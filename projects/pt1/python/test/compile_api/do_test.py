# RUN: %PYTHON %s

from dataclasses import dataclass
from typing import Optional
from torch_mlir.torchscript import do
import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return 2 * x


class ModelWithTuple(torch.nn.Module):
    def forward(self, x):
        return (2 * x,)


class ModelWithNestedTuple(torch.nn.Module):
    def forward(self, x):
        return (2 * x, [x + x])


@dataclass
class ModelOutput:
    loss: Optional[torch.FloatTensor] = None
    x: torch.FloatTensor = None
    y: torch.FloatTensor = None


class ModelWithDataclassOutput(torch.nn.Module):
    def forward(self, x):
        return ModelOutput(x=2 * x, y=x + x)


do(Model(), torch.ones(5), output_type="torch")
do(ModelWithTuple(), torch.ones(5), output_type="torch")
do(ModelWithNestedTuple(), torch.ones(5), output_type="torch")
do(ModelWithDataclassOutput(), torch.ones(5), output_type="torch")


do(Model(), torch.ones(5), output_type="tosa")
do(Model(), torch.ones(5), output_type="tosa", dtype=torch.bfloat16)
do(
    Model(),
    torch.ones(5),
    output_type="tosa",
    dtype=torch.bfloat16,
    output_prefix="out",
)
