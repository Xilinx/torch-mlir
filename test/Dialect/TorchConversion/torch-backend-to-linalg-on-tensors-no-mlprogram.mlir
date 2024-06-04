// RUN: torch-mlir-opt -p 'builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline{use-mlprogram=0})' -split-input-file %s | FileCheck %s
// RUN: torch-mlir-opt -p 'builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline{use-mlprogram=1})' -split-input-file %s | FileCheck --check-prefix=YES-CHECK %s

// CHECK-NOT: ml_program.global{{.*}}@global_seed
// YES-CHECK: ml_program.global{{.*}}@global_seed
// CHECK: func.func @torch_gemm
func.func @torch_gemm(%arg0: tensor<?x3xf32>, %arg1: tensor<3x?xf32>, %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32> {onnx.name = "gemm"}) attributes {torch.onnx_meta.opset_version = 19 : si64} {
  %0 = torch_c.from_builtin_tensor %arg0 : tensor<?x3xf32> -> !torch.vtensor<[?,3],f32>
  %1 = torch_c.from_builtin_tensor %arg1 : tensor<3x?xf32> -> !torch.vtensor<[3,?],f32>
  %2 = torch_c.from_builtin_tensor %arg2 : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %3 = torch.aten.mm %0, %1 : !torch.vtensor<[?,3],f32>, !torch.vtensor<[3,?],f32> -> !torch.vtensor<[?,?],f32>
  %4 = torch.aten.add.Tensor %3, %2, %int1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
  %5 = torch_c.to_builtin_tensor %4 : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}
