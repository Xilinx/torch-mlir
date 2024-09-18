# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import dataclasses
from enum import Enum
import inspect
from io import StringIO
import os
import sys
import tempfile
from typing import Union

import torch

from torch_mlir.passmanager import PassManager
from torch_mlir.ir import StringAttr


def get_module_name_for_debug_dump(module):
    """Gets a name suitable for a debug dump.

    The name is not guaranteed to be unique.
    """
    if not "torch.debug_module_name" in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(module.operation.attributes["torch.debug_module_name"]).value


class TorchMlirCompilerError(Exception):
    pass


def run_pipeline_with_repro_report(
    module, pipeline: str, description: str, enable_ir_printing: bool = False
):
    """Runs `pipeline` on `module`, with a nice repro report if it fails."""
    module_name = get_module_name_for_debug_dump(module)
    original_stderr = sys.stderr
    try:
        sys.stderr = StringIO()
        asm_for_error_report = module.operation.get_asm(
            large_elements_limit=10, enable_debug_info=True
        )
        # Lower module in place to make it ready for compiler backends.
        with module.context as ctx:
            # TODO(#3506): Passes can emit errors but not signal failure,
            # which causes a native assert.
            ctx.emit_error_diagnostics = True
            pm = PassManager.parse(pipeline)
            if enable_ir_printing:
                ctx.enable_multithreading(False)
                pm.enable_ir_printing()
            pm.run(module.operation)
    except Exception as e:
        # TODO: More robust.
        # - don't arbitrarily clutter up /tmp. When a test suite has many
        #   tests, this can be a big disk cost (also, /tmp/ is frequently a
        #   RAM fs, which increases worries about capacity).
        # - don't have colliding filenames (hard to do without cluttering
        #   up /tmp)
        # - if we do have have colliding filenames, writes should at least
        #   avoid being racy.
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        # Put something descriptive here even if description is empty.
        description = description or f"{module_name} compile"

        message = f"""\
            {description} failed with the following diagnostics:
            {sys.stderr.getvalue()}

            python exception: {e}

            For Torch-MLIR developers, the error can be reproduced with:
            $ torch-mlir-opt -pass-pipeline='{pipeline}' {filename}
            Add '{debug_options}' to get the IR dump for debugging purpose.
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise TorchMlirCompilerError(trimmed_message) from None
    finally:
        sys.stderr = original_stderr


class OutputType(Enum):

    # Output torch dialect in backend form. When converting from TorchDynamo,
    # this comes after some decomposition and reduce op variants passes are
    # applied to the raw torch dialect. When converting from TorchScript, this
    # comes after some cleanup passes which attempt to de-alias, decompose and infer shapes.
    # These should be roughly the same level of abstraction since those
    # steps are done within PyTorch itself when coming directly from Dynamo/FX.
    TORCH = "torch"

    # The output type contains a mix of `linalg`-on-tensors ops, `scf`, and
    # `arith` ops (and also `math` and `tm_tensor`). It can be thought of
    # as taking the `TORCH` output type and lowering it so that tensor
    # computations are done with `linalg`-on-tensors ops.
    LINALG_ON_TENSORS = "linalg-on-tensors"

    # This output type consists of `tosa` dialect ops. It can be thought of
    # as taking the `TORCH` output type and lowering it to TOSA.
    TOSA = "tosa"

    # This output type consists of `stablehlo` dialect ops. It can be thought of
    # as taking the `TORCH` output type and lowering it to StableHLO.
    STABLEHLO = "stablehlo"

    # Raw output of the JIT IR importer in the TorchScript frontend or that of
    # the FX IR importer in the TorchDynamo frontend. This is not expected to be useful
    # for end-users, but can be convenient for development or reporting bugs.
    RAW = "raw"

    @staticmethod
    def get(spec: Union[str, "OutputType"]) -> "OutputType":
        """Gets an OutputType from allowed way to specify one.

        Args:
          spec: An OutputType instance or the case-insensitive name of one of the
            enum values.
        Returns:
          An OutputType instance.
        """
        if isinstance(spec, OutputType):
            return spec
        spec = spec.upper().replace("-", "_")
        if spec not in OutputType.__members__:
            raise ValueError(
                f"For output_type= argument, expected one of: "
                f"{', '.join(OutputType.__members__.keys())}"
            )
        return OutputType[spec]


def lower_mlir_module(verbose, output_type, module):
    if verbose:
        print("\n====================")
        print("Torch Backend IR")
        print(module)

    if output_type == OutputType.TORCH:
        return module

    if output_type == OutputType.TOSA:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-tosa-backend-pipeline)",
            "Lowering Torch Backend IR -> TOSA Backend IR",
        )
        if verbose:
            print("\n====================")
            print("TOSA Backend IR")
            print(module)
        return module

    if output_type == OutputType.LINALG_ON_TENSORS:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
            "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
        )
        if verbose:
            print("\n====================")
            print("LINALG Backend IR")
            print(module)
        return module

    elif output_type == OutputType.STABLEHLO:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-stablehlo-backend-pipeline)",
            "Lowering Torch Backend IR -> StableHLO Backend IR",
        )
        if verbose:
            print("\n====================")
            print("StableHLO Backend IR")
            print(module)
        return module
    raise Exception(f"Unknown OutputType: {output_type}")


def wrap_model_return_types(model):
    """
    Wrap this model to transform return types not supported by torch_mlir
    into supported ones.
    For example, models returning a tuple of a single tensor are turned into
    models returning a single tensor instead.
    """

    def flatten(S):
        """
        Flattens a tree of list/tuples into a flat list.
        Removes list entries that are None.
        """
        if len(S) == 0:
            return S
        if isinstance(S[0], list) or isinstance(S[0], tuple):
            return list(flatten(S[0])) + list(flatten(S[1:]))
        if S[0] is None:
            return list(flatten(S[1:]))

        return list(S[:1]) + list(flatten(S[1:]))

    class Wrapper(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            ret = self.model(*args, **kwargs)

            # Torch MLIR does not support return types that are dataclasses
            # or lists or nested tuples.
            # It also does not support tuples where some elements are None.
            # Potential pytorch solution:
            #   ret, treespec = torch.utils._pytree.tree_flatten(ret)
            # but unfortunately, pytree doesn't support dataclasses
            # and it doesn't traverse base classes to see that transformer
            # outputs derive from OrderedDicts.
            # TODO: Remember the transformations done here, so we can revert
            # them outside of the model to restore the original output type.
            # See approach in make_simple_dynamo_backend.

            if dataclasses.is_dataclass(ret):
                ret = tuple(
                    [ret.__dict__[field.name] for field in dataclasses.fields(ret)]
                )

            if isinstance(ret, list) or isinstance(ret, tuple):
                ret = flatten(ret)
                if len(ret) == 1:
                    return ret[0]
                else:
                    return tuple(ret)
            return ret

    return Wrapper(model)


def map_kwargs_into_args(model, model_args, model_kwargs):
    """
    Return new_args so that
        model(*model_args, **model_kwargs)
    is equivalent to
        model(*new_args)
    """
    func_signature = inspect.signature(model.forward)
    if any(
        v.kind == inspect.Parameter.VAR_KEYWORD
        for v in func_signature.parameters.values()
        if v.name in model_kwargs
    ):
        raise TypeError("Keyword-only arguments are not supported")

    bound_arguments = func_signature.bind(*model_args, **model_kwargs)
    bound_arguments.apply_defaults()
    assert len(bound_arguments.kwargs) == 0
    new_args = bound_arguments.args

    # Remove trailings Nones from the list of arguments.
    # torch_mlir does not support passing None as argument.
    while len(new_args) > 0 and new_args[-1] is None:
        new_args = new_args[:-1]

    return new_args


def prepare_model(model, *model_args, dtype=None):
    """
    Converts the given model to an FX graph.
    WARNING: This modifies the model in-place!
    """
    model.eval()

    if dtype is not None:
        model.to(dtype)

    model = wrap_model_return_types(model)

    # Needed for models like bigbird-roberta-base that adjust their config during
    # runtime saying, e.g.
    #   Attention type 'block_sparse' is not possible ...
    #   Changing attention type to 'original_full'..."
    # Running the model once updates the config. If we trace while it updates
    # the config, torch-mlir fails with
    # error: unknown: unsupported by backend contract: module initializers
    # See https://github.com/llvm/torch-mlir/issues/2165
    golden = model(*model_args)
    return model, golden
