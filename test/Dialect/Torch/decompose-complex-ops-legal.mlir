// RUN: torch-mlir-opt -torch-decompose-complex-ops="legal-ops=aten.softmax.int" -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @torch.aten.softmax.int$cst_dim
func.func @torch.aten.softmax.int$cst_dim(%t: !torch.tensor<[2,3],f32>) -> !torch.tensor<[2,3],f32> {
  %none = torch.constant.none
  %dim = torch.constant.int 1
  // CHECK: torch.aten.softmax.int
  %ret = torch.aten.softmax.int %t, %dim, %none : !torch.tensor<[2,3],f32>, !torch.int, !torch.none -> !torch.tensor<[2,3],f32>
  return %ret : !torch.tensor<[2,3],f32>
}

// -----

func.func @torch.aten.pad.constant(%input: !torch.tensor<[2],f32>, %pads: !torch.vtensor<[2],si64>) -> !torch.tensor<[4],f32> {
  %int0 = torch.constant.int 0
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %1 = torch.aten.select.int %pads, %int0, %int0 : !torch.vtensor<[2],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  %2 = torch.aten.item %1 : !torch.vtensor<[],si64> -> !torch.int
  %pad = torch.prim.ListConstruct %2 : (!torch.int) -> !torch.list<int>
  %str = torch.constant.str "constant"
  // CHECK: torch.aten.constant_pad_nd %{{.*}}, %{{.*}}, %{{.*}} : !torch.tensor<[2],f32>, !torch.list<int>, !torch.float -> !torch.tensor<[4],f32>
  %ret = torch.aten.pad %input, %pad, %str, %float0.000000e00 : !torch.tensor<[2],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.tensor<[4],f32>
  return %ret : !torch.tensor<[4],f32>
}
