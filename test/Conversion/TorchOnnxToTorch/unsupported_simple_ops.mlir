// RUN: torch-mlir-opt <%s -split-input-file -verify-diagnostics -convert-torch-onnx-to-torch

module {
  func.func @test_argmax_no_keepdims_random_select_last_index(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,4],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    // TODO: Unsupported torch.onnx.select_last_index
    // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
    %0 = torch.operator "onnx.ArgMax"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.keepdims = 0 : si64, torch.onnx.select_last_index = 1 : si64} : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,4],si64>
    return %0 : !torch.vtensor<[2,4],si64>
  }
}

// -----
func.func @test_argmin_no_keepdims_example_select_last_index(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // TODO: Unsupported torch.onnx.select_last_index
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.ArgMin"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.keepdims = 0 : si64, torch.onnx.select_last_index = 1 : si64} : (!torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],si64>
  return %0 : !torch.vtensor<[2],si64>
}

// -----

// Less is supported starting from v13, so although this Less is legal, it will not be accepted.

func.func @test_lessv7(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
  // expected-error @+1 {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.Less"(%arg0, %arg1) { torch.onnx_meta.version = 7 : si64 } : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1>
  return %0 : !torch.vtensor<[3,4,5],i1>
}
