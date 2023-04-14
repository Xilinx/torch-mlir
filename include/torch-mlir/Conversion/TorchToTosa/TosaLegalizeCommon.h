//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZECOMMON_H
#define TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZECOMMON_H

#include "mlir/IR/PatternMatch.h" // from @llvm-project
#include "mlir/Support/LLVM.h"    // from @llvm-project

namespace mlir {
namespace tosa {

std::optional<Value>
createOneDimTfIndices(PatternRewriter &rewriter, Operation *op,
                      SmallVector<int64_t> indiceOneDimShape, int32_t dim,
                      ArrayRef<int64_t> indexShape);

std::optional<Value> convertTorchIndexToTfIndices(PatternRewriter &rewriter,
                                                   Operation *op,
                                                   Value params_value,
                                                   Value index_value,
                                                   int32_t axis);

// Lowers torch.aten.Gather operators to a sequence of TOSA ops.
// Revised from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc
std::optional<Value> convertGatherNdOp(PatternRewriter &rewriter,
                                        Operation *op, Type out_type,
                                        Value params_value,
                                        Value indices_value);

// Lowers ReduceAll to a sequence of TOSA ops.
std::optional<Value>
convertReduceAllOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceAny to a sequence of TOSA ops.
std::optional<Value>
convertReduceAnyOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceMin to a sequence of TOSA ops.
std::optional<Value>
convertReduceMinOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceMax to a sequence of TOSA ops.
std::optional<Value>
convertReduceMaxOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceProd to a sequence of TOSA ops.
std::optional<Value>
convertReduceProdOp(PatternRewriter &rewriter, Operation *op,
                    RankedTensorType output_type, Value input_value,
                    ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceSum to a sequence of TOSA ops.
std::optional<Value>
convertReduceSumOp(PatternRewriter &rewriter, Operation *op,
                   RankedTensorType output_type, Value input_value,
                   ElementsAttr axes_elems, bool keep_dims);

// Lowers ReduceMean to a sequence of TOSA ops.
std::optional<Value>
convertReduceMeanOp(PatternRewriter &rewriter, Operation *op,
                    RankedTensorType output_type, Value input_value,
                    ElementsAttr axes_elems, bool keep_dims);

} // namespace tosa
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOTOSA_TOSALEGALIZECOMMON_H
