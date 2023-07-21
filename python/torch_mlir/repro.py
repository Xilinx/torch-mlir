"""
Example:

class Model(torch.nn.Module):
        def forward(self, x):
            x = x / 2.0
            x = x + 2
            x = x * 3
            return x, x *5

model = Model()
inputs = (torch.ones(5, 4), )
out = model(*inputs)

reproduce(model, inputs, output_type="tosa", expected_error="failed to legalize")
"""


import contextlib
import io
import re
from typing import List, Optional
import torch
import torch_mlir

from torch_mlir.dynamo import _get_decomposition_table
from torch.fx.experimental.proxy_tensor import make_fx
import torch.fx as fx

from .compiler_utils import prepare_model, map_kwargs_into_args
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import (
            LinalgOnTensorsTosaBackend,
    )

# TODO: Switch to
#   from functorch.compile import minifier
# once the bug mentioned at the top of fx_minifier.py is fixed.
from .fx_minifier import minifier


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


_REs = {
    r"RuntimeError:": r"RuntimeError: ",  # change so its kept
    r"NameError:": r"NameError: ",
    r"ImportError:": r"ImportError: ",
    r"error: unknown:": r"error:",
    r"assert torch.allclose": r"Did not match accuracy",
    r'error: ["<>a-zA-Z0-9._/-]+:[0-9]+:[0-9]+: (.*)': r"error: \1",
    r".*unsupported by backend contract: tensor with unknown rank": "unsupported by backend contract: tensor with unknown rank",
    r"torch.initialize.global_slots.*": r"torch.initialize.global_slots",
    r'note: ["<>a-zA-Z0-9._/-]+:[0-9]+:[0-9]+: (.*)': r"note: \1",
    r"note: unknown:": r"note:",
    r"note: this is likely due to a missing transfer function in abstract_interp_lib_gen.py": "",
    r"%(arg)?[0-9]+": "%SSA",
    r"\[[0-9]+(,[0-9]+)*\]": r"[dims]",
}


def _reduce_error_msg(msg):
    lines = []
    for line in msg.splitlines():
        orgline = line
        for regex, replacement in _REs.items():
            line = re.sub(regex, replacement, line)
        if line != "" and line != orgline:
            lines.append(line)
    if len(lines) == 0 or (len(lines) == 1 and lines[0] == ""):
        return msg

    return ", ".join(lines).strip()


def _obtain_errror(fx_g: fx.GraphModule, inputs, output_type: str):
    """
    Runs the given module through torch_mlir and returns the error
    message produced.
    """
    # The minifer introduces functions that return a tuple with a single
    # tensor, which is not supported by torch_mlir.
    # Wrap the module to unpack those outputs.
    # torch.jit.script doesn't support *args and **kwargs as used in
    # the wrapper, so we also need to apply make_fx to the wrapped
    # model.
    # Both of those are implemented by prepare_model().
    # wrapped_g = prepare_model(model, *inputs)
    _fix_single_output_tuple(fx_g)
    with contextlib.redirect_stderr(io.StringIO()) as stderr:
        try:
            torch_mlir.compile_and_run(fx_g, inputs, output_type)
            return ""
        except Exception as e:
            return str(e) + stderr.getvalue()


def _fix_single_output_tuple(fx_g: fx.GraphModule):
    """
    torch_mlir.compile does not support modules that return a tuple of
    a single tensor.
    Change the module to return the tensor directly.
    """
    for idx, node in enumerate(fx_g.graph.nodes):
        node.idx = idx
        if node.op == "output":
            if isinstance(node.args[0], fx.Node):
                # Only one output, nothing to reduce
                return None
            if len(node.args[0]) == 1:
                node.args = (node.args[0][0], node.args[1:])
                fx_g.recompile()


def _dump_reproducer(
    fx_g: fx.GraphModule, inps: List[torch.Tensor], output_type: str, dtype
):
    _fix_single_output_tuple(fx_g)

    print("---- SNIP ----")
    print("import torch")
    print("from torch import tensor, device") # Used inside fx_g.code
    print("import torch_mlir")
    print("")

    print("class Model(torch.nn.Module):")
    print("    ".join(fx_g.code.splitlines(True)))

    print()
    print("model = Model()")
    args = ""
    for inp in inps:
        if torch.all(inp == 0):
            args += f"torch.zeros({inp.shape}, dtype={inp.dtype}), "
        elif torch.all(inp == 1):
            args += f"torch.ones({inp.shape}, dtype={inp.dtype}), "
        else:
            torch.set_printoptions(threshold=100000)
            args += f"torch.tensor({str(inp)}, dtype={inp.dtype}), "
    if dtype is not None:
        print(f"model.to({dtype})")
    print(f"inps = ({args})")
    print("golden = model(*inps)")
    print("# if you want to see the raw IR, you can print(torch_mlir.compile(model, inps, output_type='raw')")
    print(f"torch_mlir.compile_and_run(model, inps, output_type='{output_type}', golden=golden)")
    print("")
    print("---- SNIP ----")

def _reduce_inputs(inps, are_inputs_good):
    for i in range(len(inps)):
        new_inps = inps.copy()
        new_inps[i] = torch.zeros(inps[i].shape, dtype=inps[i].dtype)
        if are_inputs_good(new_inps):
            inps = new_inps
    return inps

@torch.no_grad()
def reproduce(
    model: torch.nn.Module,
    model_args,
    model_kwargs=None,
    output_type="torch",
    dtype=None,
    expected_error: Optional[str] = None,
    verbose=False,
):
    """
    Reduces the given model while ensuring that the error message seen by passing
    the model through torch_mlir.compile() doesn't change.

    When dtype is provided, calls model.to(dtype) as first step.

    This function tries to automatically determine the essential parts of the
    error message. You can also pass it explicitly via the expected_error
    parameter.
    """
    if model_kwargs is not None:
        model_args = map_kwargs_into_args(model, model_args, model_kwargs)
    model, _ = prepare_model(model, *model_args, dtype=dtype)
    fx_g = make_fx(
           model,
           decomposition_table=_get_decomposition_table())(*model_args)

    error = _obtain_errror(fx_g, model_args, output_type=output_type)
    if error == "":
        print("ERROR: torch_mlir.compile passes, nothing to reproduce")
        return

    print(f"Found error:\n{error}\nEND")

    if expected_error is None:
        expected_error = _reduce_error_msg(error)

    print(
        f"Looking for error message '{bcolors.WARNING}{expected_error}{bcolors.ENDC}'"
    )

    def module_fails(fx_g, inputs):
        error = _obtain_errror(fx_g, inputs, output_type=output_type)
        reduced_error = _reduce_error_msg(error)
        fails = expected_error in reduced_error
        if verbose:
            print(
                f"Testing graph\n{fx_g.code}\nERROR: {error}\nREDUCED_ERROR: {reduced_error}\nModule fails?: {fails}"
            )
        return fails


    def show_reproducer(fx_g: fx.GraphModule, inps: List[torch.Tensor]):
        inps = _reduce_inputs(inps, lambda inputs: module_fails(fx_g, inputs))
        _dump_reproducer(fx_g, inps, output_type, dtype)

    # Tuples are not supported by minifier
    model_args = list(model_args)
    minifier(fx_g, model_args, module_fails, dump_state=show_reproducer)
