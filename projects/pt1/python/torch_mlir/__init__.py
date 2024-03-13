# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import dataclasses
from typing import Optional, Sequence, Union, List, Dict, Tuple, Callable, Iterable
from enum import Enum
import importlib.metadata

import sys
from io import StringIO
import tempfile

from torch._functorch.compile_utils import strip_overloads
import torch
import torch.fx
from torch_mlir.dynamo import _get_decomposition_table
from torch.fx.experimental.proxy_tensor import make_fx

from .compiler_utils import run_pipeline_with_repro_report
from torch_mlir.jit_ir_importer import ClassAnnotator, ImportOptions, ModuleBuilder
from torch_mlir.jit_ir_importer.build_tools.library_generator import generate_library

from .repro import reproduce
from .compiler_utils import prepare_model, map_kwargs_into_args

class OutputType(Enum):
    """The kind of output that `torch_mlir.compile` can produce.

    In MLIR terminology, this describes the mix of dialects that will be
    produced by the conversion process.

    In user-facing API's, this type can always be passed interchangeably with an
    appropriate string specifying the output type. The allowed strings are
    the set of enum vales, allowed to be case insensitive and with `-` allowed
    in place of `_`. The `OutputType.get` static method can be used to convert
    from a string to an `OutputType` instance.
    """

    # This output type consists of `torch` dialect ops that have been converted
    # maximally to value semantics, decomposed, and shapes have been inferred.
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

    # Raw output of the JIT IR importer. This is not expected to be useful
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
            raise ValueError(f"For output_type= argument, expected one of: "
                             f"{', '.join(OutputType.__members__.keys())}")
        return OutputType[spec]


class TensorPlaceholder:
    """A class that represents a formal parameter of a given shape and dtype.

    This class can be constructed explicitly from a shape and dtype:
    ```python
    placeholder = TensorPlaceholder([3, 4], torch.float32)
    ```

    This class can also be constructed from a `torch.Tensor` which is already
    known to be a valid input to the function. In this case, a set of
    dynamic axes are allowed to be specified.
    ```python
    placeholder = TensorPlaceholder.like(torch.ones(3, 4), dynamic_axes=[1])
    # Equivalent to `TensorPlaceholder([3, -1], torch.float32)`
    ```
    """

    def __init__(self, shape: List[int], dtype: torch.dtype):
        """Create a tensor with shape `shape` and dtype `dtype`.

        Args:
            shape: The shape of the tensor. A size of `-1` indicates that the
            dimension has an unknown size.
            dtype: The dtype of the tensor.
        """
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def like(tensor: torch.Tensor, dynamic_axes: List[int] = None):
        """Create a tensor placeholder that is like the given tensor.

        Args:
            tensor: The tensor to create a placeholder for.
            dynamic_axes: A list of dynamic axes. If specified, the compiled
            module will allow those axes to be any size at runtime.
        """
        if dynamic_axes is None:
            dynamic_axes = []
        shape = []
        for i, dim in enumerate(tensor.shape):
            if i in dynamic_axes:
                shape.append(-1)
            else:
                shape.append(dim)
        return TensorPlaceholder(shape, tensor.dtype)


_example_arg = Union[TensorPlaceholder, torch.Tensor]
_example_args_for_one_method = Union[_example_arg, Sequence[_example_arg]]
_example_args = Union[_example_args_for_one_method, "ExampleArgs"]


class ExampleArgs:
    """A class representing the example arguments to an nn.Module.

    In general, an nn.Module may have multiple methods that need to be compiled.
    This requires example args for each method. This class is a lightweight
    wrapper around a dictionary that maps method names to example arguments.

    In user-facing API's, this type can always be passed interchangeably with a
    single arg or list of args, which normalizes to an ExampleArgs for just
    the `forward` method via the `ExampleArgs.get` static method.
    """

    def __init__(self):
        self._example_args = {}

    def add_method(self, method_name: str, example_args: _example_args_for_one_method):
        """Adds example args for a method.

        Args:
            method_name: The name of the method. Must have not already been
                added previously as a method.
            example_args: The example args for the method.
        Returns:
            self, for chaining.
        """
        assert method_name not in self._example_args
        self._example_args[method_name] = ExampleArgs._canonicalize_args(
            example_args)
        return self

    @staticmethod
    def get(example_args: _example_args) -> "ExampleArgs":
        """Gets an ExampleArgs from one of the permissible ways to specify one.

        Args:
          example_args: An ExampleArgs instance or a single arg or list of args.
        Returns:
          An ExampleArgs instance.
        """
        if isinstance(example_args, ExampleArgs):
            return example_args
        return ExampleArgs().add_method("forward", example_args)

    @staticmethod
    def _canonicalize_args(example_args: _example_args_for_one_method):
        """Canonicalize the args for one method into a tuple."""
        if not isinstance(example_args, Sequence):
            example_args = [example_args]
        for arg in example_args:
            if not isinstance(arg, (TensorPlaceholder, torch.Tensor)):
                raise Exception(f"Only Tensor's, TensorPlaceholder's, or sequences of "
                                f"Tensor's and TensorPlaceholder's are supported as "
                                f"example args for method inputs. "
                                f"Got '{arg}'.")
        return tuple(example_args)

    def _get_methods(self):
        return self._example_args.keys()

    def _get_for_annotation(self):
        result = {}
        for method_name, example_args in self._example_args.items():
            placeholders = []
            for arg in example_args:
                if isinstance(arg, TensorPlaceholder):
                    placeholders.append(arg)
                else:
                    assert isinstance(arg, torch.Tensor)
                    placeholders.append(TensorPlaceholder.like(arg))
            result[method_name] = placeholders
        return result

    def _get_for_tracing(
        self,
        use_tracing: bool,
        ignore_traced_shapes: bool,
    ) -> Dict[str, Tuple[_example_arg, ...]]:
        result = {}
        for method_name, example_args in self._example_args.items():
            # If we are tracing, then we need to convert any placeholders into
            # concrete values.
            if use_tracing:
                example_args_for_trace = []
                for arg in example_args:
                    if isinstance(arg, TensorPlaceholder):
                        if not ignore_traced_shapes:
                            # To avoid accidental footguns, we require
                            # `ignore_traced_shapes` to be true if we're using
                            # TensorPlaceholder's, as it falls into the same
                            # "hopefully the trace works for different inputs"
                            # bucket of concerns.
                            raise Exception(
                                "TensorPlaceholder can only be used with tracing when `ignore_traced_shapes=True`")
                        # For any dynamic dimensions, replace them with "7"
                        # arbitrarily. If a user is using dynamic dimensions with
                        # tracing, they are walking on thin ice already -- assume
                        # they know what they are doing and that their trace is
                        # correct for any specific concrete size.
                        shape = [s if s != -1 else 7 for s in arg.shape]
                        if len(shape) == 0:
                            example_args_for_trace.append(torch.tensor(1))
                        else:
                            example_args_for_trace.append(
                                torch.ones(*shape, dtype=arg.dtype))
                    else:
                        assert isinstance(arg, torch.Tensor)
                        example_args_for_trace.append(arg)
                example_args = tuple(example_args_for_trace)
            result[method_name] = example_args
        return result


# The set of ops that are considered legal for each backend.
# These are currently quite load-bearing, since different backends might be
# missing patterns for decomposed forms of certain ops.
# TODO: Tighten up the definition of these "conditionally legal for backends"
# ops in the backend contract, and move these lists somewhere deeper in the
# compiler where each backend can "own" its set of legal ops.
BACKEND_LEGAL_OPS = {
    OutputType.TOSA: ['aten.flatten.using_ints', 'aten.native_layer_norm', 'aten.linear'],
    OutputType.LINALG_ON_TENSORS: ['aten.flatten.using_ints', ],
    OutputType.STABLEHLO: [],
}


def _canon_extra_library(extra_library):
    extra_library_file_name = ""
    if len(extra_library) != 0:
        extra_library_dict = {}
        for library_func in extra_library:
            extra_library_dict[library_func.__name__] = library_func
        mlir_library = generate_library(extra_library_dict)

        extra_library_file_name = \
            tempfile.gettempdir() + "/custom_op_extra_library.mlir"
        with open(extra_library_file_name, "w") as f:
            f.write(mlir_library)
    return extra_library_file_name


def _lower_mlir_module(verbose, output_type, module):
    if verbose:
        print("\n====================")
        print("Torch Backend IR")
        print(module)

    if output_type == OutputType.TORCH:
        return module

    if output_type == OutputType.TOSA:
        run_pipeline_with_repro_report(
            module, "builtin.module(torch-backend-to-tosa-backend-pipeline)",
            "Lowering Torch Backend IR -> TOSA Backend IR")
        if verbose:
            print("\n====================")
            print("TOSA Backend IR")
            print(module)
        return module

    if output_type == OutputType.LINALG_ON_TENSORS:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
            "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR")
        if verbose:
            print("\n====================")
            print("LINALG Backend IR")
            print(module)
        return module

    elif output_type == OutputType.STABLEHLO:
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-backend-to-stablehlo-backend-pipeline)",
            "Lowering Torch Backend IR -> StableHLO Backend IR")
        if verbose:
            print("\n====================")
            print("StableHLO Backend IR")
            print(module)
        return module
    raise Exception(f"Unknown OutputType: {output_type}")


def compile(model: torch.nn.Module,
            example_args: _example_args,
            output_type: Union[str, "OutputType"] = OutputType.TORCH,
            use_tracing: bool = False,
            ignore_traced_shapes=False,
            backend_legal_ops: Optional[Sequence[str]] = None,
            extra_library: Iterable[Callable] = [],
            verbose: bool = False,
            use_make_fx: bool = False):
    """Convert a PyTorch model to MLIR.

    Args:
        model: The PyTorch model to convert.
        example_args: A list of example arguments to use when inferring the
            shapes of the arguments to `forward` method of the model.
            A single tensor is treated as a list of a single tensor.
            A TensorPlaceholder object is also allowed in the place of any
            Tensor. For models with multiple methods, an `ExampleArgs` object
            can be passed.
        output_type: The kind of output to produce. See `OutputType` for more
            details.
        use_tracing: If True, use `torch.jit.trace` to convert the model to
            JIT IR rather than `torch.jit.script`.
        ignore_traced_shapes: If True, ignore the shapes that were observed
            during tracing. This should only be used if one knows that the
            original traced program would result in the same trace (modulo
            shapes) for all shape combinations implied by any
            `TensorPlaceholder`'s used as `example_args`. Also,
            strictly-speaking, this option covers dtypes too, but we just say
            "shapes" to be succinct.
        backend_legal_ops: A list of ops that should be considered legal for
            the backend. An op that is considered legal will not be decomposed.
            This option is only valid with the `"torch"` output type.
        extra_library: List of abstract interpretation functions to splice
            into the abstract interpretation library. See
            `docs/adding_abstract_interpretation_functions.md` for more info
            on the format the functions should have.
        verbose: If true, print extra information about the conversion.

    Returns:
        An MLIR module that contains the converted model in the specified
        output type.
    """
    extra_library_file_name = _canon_extra_library(extra_library)
    output_type = OutputType.get(output_type)
    example_args = ExampleArgs.get(example_args)
    if ignore_traced_shapes and not use_tracing:
        raise Exception("`ignore_traced_shapes` requires `use_tracing`")

    # We only allow `backend_legal_ops` to be specified for the `"torch"`
    # output type because the other output types actually invoke their
    # respective backends (Linalg, TOSA, or STABLEHLO), and those backends have
    # very specific requirements about the ops which are legal.
    # See `BACKEND_LEGAL_OPS` for more details.
    if backend_legal_ops is not None:
        if output_type != OutputType.TORCH:
            raise Exception("`backend_legal_ops` is only valid with the "
                            "`torch` output type")
        backend_legal_ops = list(sorted(set(backend_legal_ops)))
    else:
        backend_legal_ops = BACKEND_LEGAL_OPS.get(output_type, [])

    if use_make_fx:
        args = example_args._get_for_tracing(use_tracing=True, ignore_traced_shapes=True)["forward"]
        model = make_fx(
           model,
           decomposition_table=_get_decomposition_table())(*args)


    # For FX-based models, automatically strip overloads.
    if isinstance(model, torch.fx.GraphModule):
        strip_overloads(model)

    # Get the model as JIT IR (TorchScript) for import.
    # TODO: Longer-term, we probably need to split `torch_mlir.compile`.
    # There should be an "acquisition" step that does
    # tracing/scripting/importing from FX/using torchdynamo.export/etc.
    # + any lowering to the backend contract. Then there should be a
    # "backend lowering" step that does the actual lowering to each
    # backend. This separation should be visible at the Python API level, and
    # we can implement a deliberately simplified API like `torch_mlir.compile`
    # on top of those building blocks.
    if isinstance(model, torch.jit.ScriptModule):
        # If the user already converted the model to JIT IR themselves, just
        # do some basic error checking, but take the model as-is.
        for method_name in example_args._get_methods():
            if not hasattr(model, method_name):
                raise Exception(
                    f"Model does not have exported method '{method_name}', "
                    f"requested in `example_args`. Consider adding "
                    f"`@torch.jit.export` to the method definition.")
        scripted = model
    elif use_tracing:
        scripted = torch.jit.trace_module(
            model,
            example_args._get_for_tracing(use_tracing, ignore_traced_shapes)
        )
    else:
        # Make sure that all the methods that the user requested get scripted.
        # By default, PyTorch only scripts the `forward` method and transitive
        # callees.
        for method_name in example_args._get_methods():
            torch.jit.export(getattr(model, method_name).__func__)
        scripted = torch.jit.script(model)
    class_annotator = ClassAnnotator()
    class_annotator.exportNone(scripted._c._type())
    for method_name, example_args in example_args._get_for_annotation().items():
        class_annotator.exportPath(scripted._c._type(), [method_name])
        annotation = [None]  # `None` is always the annotation for "self".
        for arg in example_args:
            annotation.append((arg.shape, arg.dtype, True))
        class_annotator.annotateArgs(
            scripted._c._type(), [method_name], annotation)

    mb = ModuleBuilder()
    import_options = ImportOptions()
    import_options.ignoreExistingTensorShapesAndDtypes = ignore_traced_shapes
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        # Import the TorchScript module to MLIR
        mb.import_module(scripted._c, class_annotator, import_options)
    except Exception as e:
        raise Exception(f"""
PyTorch TorchScript module -> torch-mlir Object Graph IR import failed with:
### Importer C++ Exception:
{e}
### Importer Diagnostics:
{sys.stderr.getvalue()}
""") from None
    finally:
        sys.stderr = original_stderr
    if output_type == OutputType.RAW:
        return mb.module

    option_string = "{backend-legal-ops=" + ",".join(backend_legal_ops) + \
        " extra-library=" + extra_library_file_name + "}"
    run_pipeline_with_repro_report(
        mb.module,
        f"builtin.module(torchscript-module-to-torch-backend-pipeline{option_string})",
        "Lowering TorchScript IR -> Torch Backend IR",
    )

    return _lower_mlir_module(verbose, output_type, mb.module)


def run_via_iree(module, *model_args):
    from torch.utils._pytree import tree_map
    import numpy as np
    try:
        import iree.runtime as ireert
        import iree.compiler as ireec
    except Exception as e:
        print("ERROR: Failed to import iree")
        print("pip install iree-compiler iree-runtime")
        print(e)
        sys.exit(1)

    run_pipeline_with_repro_report(
            module,
            f"builtin.module(func.func({TOSA_TO_LINALG_FUNC_PIPELINE}))",
            "Lowering TOSA backend contract to Linalg-on-Tensors backend contract")

    print("Loading inference function into IREE")
    
    # Here, mlir_module is typically going to be coming from the Torch-MLIR
    # MLIR CAPI assembly. We convert to bytecode to cross the border into the
    # IREE MLIR CAPI assembly.
    # bytecode_stream = io.BytesIO()
    # module.operation.write_bytecode(bytecode_stream)
    # bytecode = bytecode_stream.getvalue()
    bytecode = module.operation.get_asm()
    iree_vmfb = ireec.compile_str(bytecode,
                             target_backends=["llvm-cpu"],
                             input_type=ireec.InputType.TM_TENSOR)
    
    config = ireert.Config(driver_name="local-sync")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, iree_vmfb)
    ctx.add_vm_module(vm_module)

    class IREEInvoker:
        """A wrapper around an IREE module that provides a Pythonic interface.
        
        Specifically, this adapts `module.forward(...)` and similar calls into
        lower-level calls into the functions in the IREE module, and also converts
        between the IREE and Torch types.
        """

        def __init__(self, iree_module):
            self._iree_module = iree_module
            self.device = iree_module._context.config.device

        def __getattr__(self, function_name: str):
            def invoke(*args):
                def wrap(x):
                    if isinstance(x, torch.Tensor):
                        return ireert.asdevicearray(self.device, x)
                    return x
                def unwrap(x):
                    if isinstance(x, ireert.DeviceArray):
                        return torch.from_numpy(np.asarray(x).copy())
                    return x
                # TODO: Investigate how to share CUDA arrays between IREE and Torch.
                iree_args = tree_map(wrap, args)
                result = self._iree_module[function_name](*iree_args)
                # TODO: Investigate why a copy is needed here.
                # Without the copy, certain sets of tests, when run together, will
                # cause a segfault when the process is exiting.
                # It seems to be related to Torch attempting to free a Numpy array
                # that is backed by IREE memory, resulting in
                # iree_hal_buffer_view_release reading from a null pointer.
                return tree_map(unwrap, result)
            return invoke
    
    invoker = IREEInvoker(ctx.modules.module)

    print("Running inference on IREE")
    return invoker.forward(*model_args)

def run_and_compare(module, model_args, golden):
    output = run_via_iree(module, *model_args)
    if not isinstance(output, tuple):
        golden = (golden, )
        output = (output, )

    assert len(output) == len(golden)
    for output_el, golden_el in zip(output, golden):
        rel_err = torch.max((output_el - golden_el)/torch.abs(golden_el))
        print("Relative error: ", rel_err)
        assert torch.allclose(output_el, golden_el, rtol=1e-2), "Accuracy issue"
    return output

def compile_and_run(model, model_args, output_type, golden = None):
    compile_output_type = output_type
    if compile_output_type == "check-tosa":
        compile_output_type = "tosa"

    if compile_output_type == "run-tosa":
        compile_output_type = "tosa"

    module = compile(model,model_args,output_type=compile_output_type, use_make_fx=True)

    if output_type == "run-tosa":
        if golden is None:
            golden = model(*model_args)
        return run_and_compare(module, model_args, golden)
    elif output_type == "check-tosa":
        # TOSA lacks a bunch of verifiers.
        # Our best way to find issues in the TOSA IR is to try to lower to Linalg
        backend = LinalgOnTensorsTosaBackend()
        backend.compile(module)

    return module

@torch.no_grad()
def do(model: torch.nn.Module,
       *model_args,
       output_type: Union[str, "OutputType"] = OutputType.TORCH,
       dtype = None,
       output_prefix: Optional[str] = None,
       verbose: bool = True,
       **model_kwargs,
       ):
    """
    Converts the given model to torch/tosa.
    WARNING: This modifies the model in-place!
    """

    model_args = map_kwargs_into_args(model, model_args, model_kwargs)

    if verbose:
        try:
            version = importlib.metadata.version('torch-mlir')
        except importlib.metadata.PackageNotFoundError:
            version = "dev"
        print(f"Using torch-mlir {version}")

    model, golden = prepare_model(model, *model_args, dtype=dtype)

    compile_output_type = output_type
    if compile_output_type in ("check-tosa", "run-tosa"):
        compile_output_type = "tosa"

    module = compile(model,model_args,output_type=compile_output_type, use_make_fx=True)
    if output_type == "run-tosa":
        output = run_via_iree(module, *model_args)
        if not isinstance(output, tuple):
            golden = (golden, )
            output = (output, )

        assert len(output) == len(golden)
        for output_el, golden_el in zip(output, golden):
            rel_err = torch.max((output_el - golden_el)/torch.abs(golden_el))
            print("Relative error: ", rel_err)
            assert torch.allclose(output_el, golden_el, rtol=1e-2), "Accuracy issue"
        return output

    if output_prefix is not None:
        prefix = f"{output_prefix}.{output_type}"
        if dtype is not None:
            assert dtype == torch.bfloat16
            prefix += ".bf16"

        if verbose:
            print(f"Writing output files with prefix {prefix}")
        with open(f"{prefix}.full.mlir", "w+") as f:
            f.write(module.operation.get_asm())
        with open(f"{prefix}.mlir", "w+") as f:
            f.write(module.operation.get_asm(large_elements_limit=10))

    return module
