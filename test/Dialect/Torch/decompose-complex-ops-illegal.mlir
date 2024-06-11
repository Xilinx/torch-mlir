// RUN: torch-mlir-opt -torch-decompose-complex-ops -split-input-file %s | FileCheck %s

func.func @torch.aten.pad.reflect(%input: !torch.tensor<[2],f32>, %pads: !torch.vtensor<[2],si64>) -> !torch.tensor<[4],f32> {
  %int0 = torch.constant.int 0
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %1 = torch.aten.select.int %pads, %int0, %int0 : !torch.vtensor<[2],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  %2 = torch.aten.item %1 : !torch.vtensor<[],si64> -> !torch.int
  %pad = torch.prim.ListConstruct %2 : (!torch.int) -> !torch.list<int>
  %str = torch.constant.str "reflect"
  // CHECK: torch.aten.pad %{{.*}} %{{.*}} %{{.*}} %{{.*}} : !torch.tensor<[2],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.tensor<[4],f32>
  %ret = torch.aten.pad %input, %pad, %str, %float0.000000e00 : !torch.tensor<[2],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.tensor<[4],f32>
  return %ret : !torch.tensor<[4],f32>
}

// -----

func.func @torch.aten.pad.edge(%input: !torch.tensor<[2],f32>, %pads: !torch.vtensor<[2],si64>) -> !torch.tensor<[4],f32> {
  %int0 = torch.constant.int 0
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %1 = torch.aten.select.int %pads, %int0, %int0 : !torch.vtensor<[2],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  %2 = torch.aten.item %1 : !torch.vtensor<[],si64> -> !torch.int
  %pad = torch.prim.ListConstruct %2 : (!torch.int) -> !torch.list<int>
  %str = torch.constant.str "edge"
  // CHECK: torch.aten.pad %{{.*}} %{{.*}} %{{.*}} %{{.*}} : !torch.tensor<[2],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.tensor<[4],f32>
  %ret = torch.aten.pad %input, %pad, %str, %float0.000000e00 : !torch.tensor<[2],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.tensor<[4],f32>
  return %ret : !torch.tensor<[4],f32>
}

// -----

func.func @torch.aten.pad.wrap(%input: !torch.tensor<[2],f32>, %pads: !torch.vtensor<[2],si64>) -> !torch.tensor<[4],f32> {
  %int0 = torch.constant.int 0
  %float0.000000e00 = torch.constant.float 0.000000e+00
  %1 = torch.aten.select.int %pads, %int0, %int0 : !torch.vtensor<[2],si64>, !torch.int, !torch.int -> !torch.vtensor<[],si64>
  %2 = torch.aten.item %1 : !torch.vtensor<[],si64> -> !torch.int
  %pad = torch.prim.ListConstruct %2 : (!torch.int) -> !torch.list<int>
  %str = torch.constant.str "wrap"
  // CHECK: torch.aten.pad %{{.*}} %{{.*}} %{{.*}} %{{.*}} : !torch.tensor<[2],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.tensor<[4],f32>
  %ret = torch.aten.pad %input, %pad, %str, %float0.000000e00 : !torch.tensor<[2],f32>, !torch.list<int>, !torch.str, !torch.float -> !torch.tensor<[4],f32>
  return %ret : !torch.tensor<[4],f32>
}