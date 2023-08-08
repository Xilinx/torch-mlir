// RUN: torch-mlir-opt <%s -convert-torch-to-tosa=enable-custom-op-conversion=true -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.gelu$basic(
// CHECK-SAME:                                     %[[VAL_0:.*]]: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
// CHECK:           %[[VAL_1:.*]] = torch_c.to_builtin_tensor %[[VAL_0]] : !torch.vtensor<[?,?,?],f32> -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_2:.*]] = torch.constant.str "none"
// CHECK:           %[[VAL_3:.*]] = "tosa.custom"(%[[VAL_1]]) <{config = "UNDEF", identifier = "Gelu", implementation_attrs = "UNDEF"}> {approx = "none"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           %[[VAL_4:.*]] = torch_c.from_builtin_tensor %[[VAL_3]] : tensor<?x?x?xf32> -> !torch.vtensor<[?,?,?],f32>
// CHECK:           return %[[VAL_4]] : !torch.vtensor<[?,?,?],f32>
// CHECK:         }
func.func @torch.aten.gelu$basic(%arg0: !torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[?,?,?],f32> {
  %str = torch.constant.str "none"
  %0 = torch.aten.gelu %arg0, %str : !torch.vtensor<[?,?,?],f32>, !torch.str -> !torch.vtensor<[?,?,?],f32>
  return %0 : !torch.vtensor<[?,?,?],f32>
}
