// RUN: torch-mlir-opt -torch-decompose-complex-ops -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @matmul_no_decompose
// CHECK:           torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
func.func @matmul_no_decompose(%arg0: !torch.vtensor<[?,?,?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}


// -----

// CHECK-LABEL:   func.func @matmul_decompose_2d
// CHECK:           torch.aten.mm %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.tensor
func.func @matmul_decompose_2d(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL:   func.func @matmul_decompose_3d(
// CHECK:           torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
func.func @matmul_decompose_3d(%arg0: !torch.vtensor<[?,?,?],f32>, %arg1: !torch.vtensor<[?,?,?],f32>) -> !torch.tensor {
  %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,?,?],f32>, !torch.vtensor<[?,?,?],f32> -> !torch.tensor
  return %0 : !torch.tensor
}

// -----
// CHECK-LABEL:   func @torch.aten.adaptive_avg_pool2d$non_unit_output_size(
// CHECK-SAME:                  %[[SELF:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK-DAG:       %[[CST0:.*]] = torch.constant.int 0
// CHECK-DAG:       %[[CST1:.*]] = torch.constant.int 1
// CHECK-DAG:       %[[CST2:.*]] = torch.constant.int 2
// CHECK-DAG:       %[[CST3:.*]] = torch.constant.int 3
// CHECK-DAG:       %[[CST6:.*]] = torch.constant.int 6
// CHECK-DAG:       %[[CST7:.*]] = torch.constant.int 7
// CHECK-DAG:       %[[FALSE:.*]] = torch.constant.bool false
// CHECK-DAG:       %[[TRUE:.*]] = torch.constant.bool true
// CHECK-DAG:       %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[SELF]], %[[CST2]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[DIM3:.*]] = torch.aten.size.int %[[SELF]], %[[CST3]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[COND1:.*]] = torch.aten.eq.int %[[DIM2]], %[[CST7]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           torch.runtime.assert %[[COND1]], "unimplemented: only support cases where input and output size are equal for non-unit output size"
// CHECK:           %[[T1:.*]] = torch.aten.sub.int %[[DIM2]], %[[CST6]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[COND2:.*]] = torch.aten.eq.int %[[DIM3]], %[[CST7]] : !torch.int, !torch.int -> !torch.bool
// CHECK:           torch.runtime.assert %[[COND2]], "unimplemented: only support cases where input and output size are equal for non-unit output size"
// CHECK:           %[[T2:.*]] = torch.aten.sub.int %[[DIM3]], %[[CST6]] : !torch.int, !torch.int -> !torch.int
// CHECK:           %[[KERNEL_SIZE:.*]] = torch.prim.ListConstruct %[[T1]], %[[T2]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[STRIDE:.*]] = torch.prim.ListConstruct %[[CST1]], %[[CST1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[PADDING:.*]]  = torch.prim.ListConstruct %[[CST0]], %[[CST0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[AVG_POOL:.*]] = torch.aten.avg_pool2d %[[SELF]], %[[KERNEL_SIZE]], %[[STRIDE]], %[[PADDING]], %[[FALSE]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.adaptive_avg_pool2d$non_unit_output_size(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int7 = torch.constant.int 7
  %output_size = torch.prim.ListConstruct %int7, %int7 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.adaptive_avg_pool2d %arg0, %output_size : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.adaptive_avg_pool2d$unit_output_size(
// CHECK-SAME:                                                               %[[SELF:.*]]: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
// CHECK-DAG:       %[[CST0:.*]] = torch.constant.int 0
// CHECK-DAG:       %[[CST1:.*]] = torch.constant.int 1
// CHECK-DAG:       %[[CST2:.*]] = torch.constant.int 2
// CHECK-DAG:       %[[CST3:.*]] = torch.constant.int 3
// CHECK-DAG:       %[[FALSE:.*]] = torch.constant.bool false
// CHECK-DAG:       %[[TRUE:.*]] = torch.constant.bool true
// CHECK-DAG:       %[[NONE:.*]] = torch.constant.none
// CHECK:           %[[DIM2:.*]] = torch.aten.size.int %[[SELF]], %[[CST2]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[DIM3:.*]] = torch.aten.size.int %[[SELF]], %[[CST3]] : !torch.vtensor<[?,?,?,?],f32>, !torch.int -> !torch.int
// CHECK:           %[[KERNEL_SIZE:.*]] = torch.prim.ListConstruct %[[DIM2]], %[[DIM3]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[STRIDE:.*]] = torch.prim.ListConstruct %[[CST1]], %[[CST1]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[PADDING:.*]] = torch.prim.ListConstruct %[[CST0]], %[[CST0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK:           %[[AVG_POOL:.*]] = torch.aten.avg_pool2d %[[SELF]], %[[KERNEL_SIZE]], %[[STRIDE]], %[[PADDING]], %[[FALSE]], %[[TRUE]], %[[NONE]] : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[?,?,?,?],f32>
func.func @torch.aten.adaptive_avg_pool2d$unit_output_size(%arg0: !torch.vtensor<[?,?,?,?],f32>) -> !torch.vtensor<[?,?,?,?],f32> {
  %int1 = torch.constant.int 1
  %output_size = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.adaptive_avg_pool2d %arg0, %output_size : !torch.vtensor<[?,?,?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?,?],f32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.acos$int_type(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !torch.vtensor<[2,2],si32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: !torch.vtensor<[2,2],si32>) -> !torch.vtensor<[2,2],si32> {
// CHECK:           %[[VAL_2:.*]] = torch.constant.int 1
// CHECK:           %[[VAL_3:.*]] = torch.aten.add.Scalar %[[VAL_0]], %[[VAL_2]], %[[VAL_2]] : !torch.vtensor<[2,2],si32>, !torch.int, !torch.int -> !torch.vtensor<[2,2],si32>
// CHECK:           %[[VAL_4:.*]] = torch.aten.neg %[[VAL_0]] : !torch.vtensor<[2,2],si32> -> !torch.vtensor<[2,2],si32>
// CHECK:           %[[VAL_5:.*]] = torch.aten.add.Scalar %[[VAL_4]], %[[VAL_2]], %[[VAL_2]] : !torch.vtensor<[2,2],si32>, !torch.int, !torch.int -> !torch.vtensor<[2,2],si32>
// CHECK:           %[[VAL_6:.*]] = torch.aten.mul.Tensor %[[VAL_3]], %[[VAL_5]] : !torch.vtensor<[2,2],si32>, !torch.vtensor<[2,2],si32> -> !torch.vtensor<[2,2],si32>
// CHECK:           %[[VAL_7:.*]] = torch.aten.sqrt %[[VAL_6]] : !torch.vtensor<[2,2],si32> -> !torch.vtensor<[2,2],si32>
// CHECK:           %[[VAL_8:.*]] = torch.aten.atan2 %[[VAL_7]], %[[VAL_0]] : !torch.vtensor<[2,2],si32>, !torch.vtensor<[2,2],si32> -> !torch.vtensor<[2,2],si32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[2,2],si32>
// CHECK:         }

func.func @torch.aten.acos$int_type(%arg0: !torch.vtensor<[2, 2],si32>, %arg1: !torch.vtensor<[2, 2],si32>) -> !torch.vtensor<[2, 2],si32> {
  %0 = torch.aten.acos %arg0 : !torch.vtensor<[2, 2],si32> -> !torch.vtensor<[2, 2],si32>
  return %0 : !torch.vtensor<[2, 2],si32>
}

// -----

// CHECK-LABEL:   func.func @torch.aten.acos$float_type(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !torch.vtensor<[2,2],f32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
// CHECK:           %[[VAL_2:.*]] = torch.constant.float 1.000000e+00
// CHECK:           %[[VAL_3:.*]] = torch.aten.add.Scalar %[[VAL_0]], %[[VAL_2]], %[[VAL_2]] : !torch.vtensor<[2,2],f32>, !torch.float, !torch.float -> !torch.vtensor<[2,2],f32>
// CHECK:           %[[VAL_4:.*]] = torch.aten.neg %[[VAL_0]] : !torch.vtensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK:           %[[VAL_5:.*]] = torch.aten.add.Scalar %[[VAL_4]], %[[VAL_2]], %[[VAL_2]] : !torch.vtensor<[2,2],f32>, !torch.float, !torch.float -> !torch.vtensor<[2,2],f32>
// CHECK:           %[[VAL_6:.*]] = torch.aten.mul.Tensor %[[VAL_3]], %[[VAL_5]] : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK:           %[[VAL_7:.*]] = torch.aten.sqrt %[[VAL_6]] : !torch.vtensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK:           %[[VAL_8:.*]] = torch.aten.atan2 %[[VAL_7]], %[[VAL_0]] : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
// CHECK:           return %[[VAL_8]] : !torch.vtensor<[2,2],f32>
// CHECK:         }
func.func @torch.aten.acos$float_type(%arg0: !torch.vtensor<[2, 2],f32>, %arg1: !torch.vtensor<[2, 2],f32>) -> !torch.vtensor<[2, 2],f32> {
  %0 = torch.aten.acos %arg0 : !torch.vtensor<[2, 2],f32> -> !torch.vtensor<[2, 2],f32>
  return %0 : !torch.vtensor<[2, 2],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.type_as$basic(
// CHECK-SAME:                                %[[ARG_0:.*]]: !torch.tensor, %[[ARG_1:.*]]: !torch.tensor) -> !torch.tensor {
// CHECK:          %[[FALSE:.*]] = torch.constant.bool false
// CHECK:          %[[NONE:.*]] = torch.constant.none
// CHECK:          %[[DTYPE:.*]] = torch.prim.dtype %[[ARG_1]] : !torch.tensor -> !torch.int
// CHECK:          %[[VAR:.*]] = torch.aten.to.dtype %[[ARG_0]], %[[DTYPE]], %[[FALSE]], %[[FALSE]], %[[NONE]] : !torch.tensor, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.tensor
// CHECK:          return %[[VAR]] : !torch.tensor
func.func @torch.aten.type_as$basic(%arg0: !torch.tensor, %arg1: !torch.tensor) -> !torch.tensor {
  %0 = torch.aten.type_as %arg0, %arg1 : !torch.tensor, !torch.tensor -> !torch.tensor
  return %0 : !torch.tensor
}

// -----

// CHECK-LABEL:   func.func @torch.aten.type_as$fold(
// CHECK-SAME:                                 %[[ARG_0:.*]]: !torch.tensor<[?],f16>, %[[ARG_1:.*]]: !torch.tensor<[?,?],f16>) -> !torch.tensor<[?],f16> {
// CHECK:           return %[[ARG_0]] : !torch.tensor<[?],f16>
func.func @torch.aten.type_as$fold(%arg0: !torch.tensor<[?], f16>, %arg1: !torch.tensor<[?,?],f16>) -> !torch.tensor<[?],f16> {
  %0 = torch.aten.type_as %arg0, %arg1 : !torch.tensor<[?], f16>, !torch.tensor<[?,?],f16> -> !torch.tensor<[?], f16>
  return %0 : !torch.tensor<[?], f16>
}
