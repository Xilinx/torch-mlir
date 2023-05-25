# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from io import StringIO
import os
import sys
import tempfile

from torch_mlir.passmanager import PassManager
from torch_mlir.ir import StringAttr
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
import torch

def get_module_name_for_debug_dump(module):
    """Gets a name suitable for a debug dump.

    The name is not guaranteed to be unique.
    """
    if not "torch.debug_module_name" in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(module.operation.attributes["torch.debug_module_name"]).value


class TorchMlirCompilerError(Exception):
    def __init__(self, value: str):
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return self.value


def run_pipeline_with_repro_report(module,
                                   pipeline: str,
                                   description: str):
    """Runs `pipeline` on `module`, with a nice repro report if it fails."""
    module_name = get_module_name_for_debug_dump(module)
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        asm_for_error_report = module.operation.get_asm(
            large_elements_limit=10, enable_debug_info=True)
        # Lower module in place to make it ready for compiler backends.
        with module.context:
            pm = PassManager.parse(pipeline)
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
        with open(filename, 'w') as f:
            f.write(asm_for_error_report)
        debug_options="-mlir-print-ir-after-all -mlir-disable-threading"
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
        trimmed_message = '\n'.join([m.lstrip() for m in message.split('\n')])
        raise TorchMlirCompilerError(trimmed_message) from None
    finally:
        sys.stderr = original_stderr

def model_to_fxgraph(model, *model_args, dtype = None, **model_kwargs):
    """
    Converts the given model to an FX graph.
    WARNING: This modifies the model in-place!
    """
        
    assert len(model_kwargs) == 0, "model_kwargs are not supported yet"

    model.eval()

    model(*model_args, **model_kwargs)

    def flatten(S):
        if len(S) == 0:
            return S
        if isinstance(S[0], list) or isinstance(S[0], tuple):
            return list(flatten(S[0])) + list(flatten(S[1:]))
        return list(S[:1]) + list(flatten(S[1:]))

    class Wrapper(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            ret = self.model(*args, **kwargs)
            
            if isinstance(ret, list) or isinstance(ret, tuple):
                ret = flatten(ret)
                if len(ret) == 1:
                    return ret[0]
                else:
                    return tuple(ret)
            return ret

    model = Wrapper(model)

    if dtype is not None:
        model.to(dtype)

    fx_g = make_fx(
           model,
           # sometimes there are decompositions for unsupported ops available.
           # we don't currently know where these are listed, but just try adding
           # the op here and see if the previously unsupported op is no longer
           # produced (you should then see the decomposition in the IR)
           decomposition_table=get_decompositions(
            [
            torch.ops.aten.cumsum,
            torch.ops.aten.embedding_dense_backward,
            torch.ops.aten.native_layer_norm_backward,
            torch.ops.aten.slice_backward,
            torch.ops.aten.select_backward,
            torch.ops.aten.norm.ScalarOpt_dim,
            torch.ops.aten.native_group_norm,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split_with_sizes,
            ]
             ),)(*model_args)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()
    return fx_g
