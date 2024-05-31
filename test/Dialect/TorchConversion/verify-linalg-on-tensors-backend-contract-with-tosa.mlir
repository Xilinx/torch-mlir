// RUN: torch-mlir-opt -p 'builtin.module(torch-verify-linalg-on-tensors-backend-contract{allow-tosa-dialect=1})' -split-input-file -verify-diagnostics -allow-unregistered-dialect %s | FileCheck %s

// CHECK: func.func @tosa
func.func @tosa(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %1 = tosa.abs %arg0 : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
