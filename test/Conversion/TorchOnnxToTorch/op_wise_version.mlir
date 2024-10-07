// RUN: torch-mlir-opt <%s --split-input-file -convert-torch-onnx-to-torch | FileCheck %s

// CHECK-LABEL: @test_quantizelinear_opset_16_op_19
func.func @test_quantizelinear_opset_16_op_19(%arg0: !torch.vtensor<[6],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 16 : si64} {
  // CHECK-NOT: torch.operator
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) {torch.onnx_meta.version = 19 : si64} : (!torch.vtensor<[6],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8>
  return %0 : !torch.vtensor<[6],si8>
}

// -----

// CHECK-LABEL: @test_quantizelinear_no_opset_op_19
func.func @test_quantizelinear_no_opset_op_19(%arg0: !torch.vtensor<[6],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8> attributes {torch.onnx_meta.ir_version = 9 : si64} {
  // CHECK-NOT: torch.operator
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) {torch.onnx_meta.version = 19 : si64} : (!torch.vtensor<[6],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8>
  return %0 : !torch.vtensor<[6],si8>
}
