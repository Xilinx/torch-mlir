// RUN: torch-mlir-opt <%s --split-input-file -verify-diagnostics -convert-torch-onnx-to-torch


func.func @test_quantizelinear_reject_block_size(
    %arg0: !torch.vtensor<[6],f32>,
    %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 23 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2)
      {torch.onnx.block_size = 32 : si64}
      : (!torch.vtensor<[6],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>)
      -> !torch.vtensor<[6],si8>
  return %0 : !torch.vtensor<[6],si8>
}

// -----

func.func @test_quantizelinear_reject_output_dtype(
    %arg0: !torch.vtensor<[6],f32>,
    %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 23 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2)
      {torch.onnx.output_dtype = 3 : si64}
      : (!torch.vtensor<[6],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>)
      -> !torch.vtensor<[6],si8>
  return %0 : !torch.vtensor<[6],si8>
}

// -----

func.func @test_quantizelinear_reject_precision(
    %arg0: !torch.vtensor<[6],f32>,
    %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 23 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2)
      {torch.onnx.precision = 1 : si64}
      : (!torch.vtensor<[6],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>)
      -> !torch.vtensor<[6],si8>
  return %0 : !torch.vtensor<[6],si8>
}

// -----

func.func @test_quantizelinear_reject_saturate(
    %arg0: !torch.vtensor<[6],f32>,
    %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],si8>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 23 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2)
      {torch.onnx.saturate = 0 : si64}
      : (!torch.vtensor<[6],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>)
      -> !torch.vtensor<[6],si8>
  return %0 : !torch.vtensor<[6],si8>
}

// -----

func.func @test_dequantizelinear_reject_block_size(
    %arg0: !torch.vtensor<[6],si8>,
    %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],f32>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 23 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.DequantizeLinear"(%arg0, %arg1, %arg2)
      {torch.onnx.block_size = 32 : si64}
      : (!torch.vtensor<[6],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>)
      -> !torch.vtensor<[6],f32>
  return %0 : !torch.vtensor<[6],f32>
}

// -----

func.func @test_dequantizelinear_reject_output_dtype(
    %arg0: !torch.vtensor<[6],si8>,
    %arg1: !torch.vtensor<[],f32>,
    %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[6],f32>
    attributes {torch.onnx_meta.ir_version = 10 : si64,
                torch.onnx_meta.opset_version = 23 : si64} {
  // expected-error @below {{failed to legalize operation 'torch.operator'}}
  %0 = torch.operator "onnx.DequantizeLinear"(%arg0, %arg1, %arg2)
      {torch.onnx.output_dtype = 1 : si64}
      : (!torch.vtensor<[6],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>)
      -> !torch.vtensor<[6],f32>
  return %0 : !torch.vtensor<[6],f32>
}
