//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"       // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h" // from @llvm-project
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeCommon.h"

namespace mlir {
namespace tosa {

// Create a TOSA rescale op from input framework tensor, zero points and
// rounding mode
Value buildRescale(PatternRewriter &rewriter, Operation *op,
                   ShapedType output_type, Value input_val, double scale,
                   int64_t input_zp, int64_t output_zp, bool double_round,
                   bool scale32) {
  int32_t multiplier;
  int32_t shift;

  int32_t scale_width = scale32 ? 32 : 16;

  computeMultiplierAndShift(scale, multiplier, shift, scale_width);

  auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
      rewriter, op->getLoc(), output_type, input_val,
      rewriter.getI32IntegerAttr(static_cast<int32_t>(input_zp)),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(output_zp)),
      rewriter.getDenseI32ArrayAttr({multiplier}),
      rewriter.getDenseI32ArrayAttr({shift}), rewriter.getBoolAttr(scale32),
      rewriter.getBoolAttr(double_round), rewriter.getBoolAttr(false));

  return rescale_op.getResult();
}

// Creates TOSA rescale op with int32 output
Value buildRescaleToInt32(PatternRewriter &rewriter, Operation *op,
                          Value input_val, double input_scale,
                          int64_t input_zp) {
  // Output is always int32 type
  auto input_type = input_val.getType().dyn_cast<mlir::ShapedType>();
  assert(input_type);
  auto output_type = input_type.clone(rewriter.getI32Type());

  return buildRescale(rewriter, op, output_type, input_val, input_scale,
                      input_zp, 0, false, true);
}

// Creates a TOSA rescale op based on conv2d parameters.
Value buildRescaleOpConvOutput(PatternRewriter &rewriter, Operation *op,
                               Value conv_val, ShapedType input_type,
                               ShapedType weight_type, ShapedType output_type) {
  auto input_qtype =
      input_type.getElementType().dyn_cast<mlir::quant::UniformQuantizedType>();
  auto output_qtype = output_type.getElementType()
                          .dyn_cast<mlir::quant::UniformQuantizedType>();

  double input_scale = input_qtype.getScale();

  int64_t output_zp = output_qtype.getZeroPoint();
  double output_scale = output_qtype.getScale();

  bool scale32 = isScale32(output_qtype);
  int32_t scale_width = scale32 ? 32 : 16;

  if (auto weight_per_tensor_qtype =
          weight_type.getElementType()
              .dyn_cast<mlir::quant::UniformQuantizedType>()) {
    // Per-tensor quantization
    double weight_scale = weight_per_tensor_qtype.getScale();

    int32_t multiplier;
    int32_t shift;

    double op_tensor_scale = (input_scale * weight_scale) / output_scale;

    computeMultiplierAndShift(op_tensor_scale, multiplier, shift, scale_width);

    auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
        rewriter, op->getLoc(), output_type, conv_val,
        rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(output_zp),
        rewriter.getDenseI32ArrayAttr({multiplier}),
        rewriter.getDenseI32ArrayAttr({shift}), rewriter.getBoolAttr(scale32),
        rewriter.getBoolAttr(true), rewriter.getBoolAttr(false));

    return rescale_op.getResult();

  } else if (auto weight_per_channel_qtype =
                 weight_type.getElementType()
                     .dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
    // Per-channel quantization
    SmallVector<int32_t> multiplier_arr;
    SmallVector<int32_t> shift_arr;

    SmallVector<double> weight_scale_arr(
        weight_per_channel_qtype.getScales().begin(),
        weight_per_channel_qtype.getScales().end());

    int64_t output_zp = output_qtype.getZeroPoint();
    double output_scale = output_qtype.getScale();

    for (double weight_scale : weight_scale_arr) {
      int32_t multiplier;
      int32_t shift;

      double op_channel_scale = (input_scale * weight_scale) / output_scale;

      computeMultiplierAndShift(op_channel_scale, multiplier, shift,
                                scale_width);

      multiplier_arr.push_back(multiplier);
      shift_arr.push_back(shift);
    }

    auto rescale_op = CreateOpAndInfer<tosa::RescaleOp>(
        rewriter, op->getLoc(), output_type, conv_val,
        rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(output_zp),
        rewriter.getDenseI32ArrayAttr(multiplier_arr),
        rewriter.getDenseI32ArrayAttr(shift_arr), rewriter.getBoolAttr(scale32),
        rewriter.getBoolAttr(true), rewriter.getBoolAttr(true));

    return rescale_op.getResult();

  } else {
    op->emitOpError("buildConvRescaleOp: unknown weight quantized type");
    return nullptr;
  }
}

Value buildSlice(PatternRewriter &rewriter, Value &input,
                 llvm::ArrayRef<int64_t> start, llvm::ArrayRef<int64_t> size) {
  assert(start.size() == size.size() &&
         "Start and Size must have the same size");
  return tosa::CreateOpAndInfer<mlir::tosa::SliceOp>(
      rewriter, input.getLoc(),
      RankedTensorType::get(
          llvm::SmallVector<int64_t, 4>(size.size(), ShapedType::kDynamic),
          input.getType().cast<ShapedType>().getElementType()),
      input, start, size);
}

// Check if scale32 mode is used for given output_element_type
bool isScale32(mlir::quant::UniformQuantizedType output_element_type) {
  return (output_element_type.getStorageTypeIntegralWidth() == 8);
}

// Create a 32-bit float constant operator from a float
Value getTosaConstTensorSingleF32(PatternRewriter &rewriter, Operation *op,
                                  float val) {
  auto const_type = RankedTensorType::get({}, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, val);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  return const_op.getResult();
}

// Create a zero constant tensor of the desired type and shape.
std::optional<Value> getZerosLikeTensor(PatternRewriter &rewriter,
                                        Operation *op, Type type) {
  RankedTensorType resultType = type.dyn_cast<RankedTensorType>();

  if (!resultType) {
    (void)rewriter.notifyMatchFailure(op, "not ranked tensor type");
    return std::nullopt;
  }

  auto resultShape = resultType.getShape();
  ShapedType zeroType =
      RankedTensorType::get(resultShape, resultType.getElementType());
  Attribute zeroAttr = rewriter.getZeroAttr(zeroType);

  return CreateOpAndInfer<tosa::ConstOp>(rewriter, op->getLoc(), zeroType,
                                         zeroAttr.cast<ElementsAttr>())
      .getResult();
}

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
template <typename T>
std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape, std::optional<Type> dtype) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements && vec.size() != 1) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto width = sizeof(T) * 8;
  if constexpr(std::is_same_v<T, bool>)
    width = 1;

  auto const_type =
      RankedTensorType::get(shape, rewriter.getIntegerType(width));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);

  if (dtype) {
   return rewriter.createOrFold<tosa::CastOp>(
        op->getLoc(), RankedTensorType::get(shape, *dtype), const_op);
  }
  return const_op.getResult();
}

// Template specialization for APInt
template <>
std::optional<Value> getConstTensor<APInt>(PatternRewriter &rewriter,
                                           Operation *op, ArrayRef<APInt> vec,
                                           ArrayRef<int64_t> shape, std::optional<Type> dtype) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements && vec.size() != 1) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  
  if (dtype) {
   return rewriter.createOrFold<tosa::CastOp>(
        op->getLoc(), RankedTensorType::get(shape, *dtype), const_op);
  }
  return const_op.getResult();
}

// Template specialization for float
template <>
std::optional<Value> getConstTensor<float>(PatternRewriter &rewriter,
                                           Operation *op, ArrayRef<float> vec,
                                           ArrayRef<int64_t> shape, std::optional<Type> dtype) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements && vec.size() != 1) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);
  
  if (dtype) {
   return rewriter.createOrFold<tosa::CastOp>(
        op->getLoc(), RankedTensorType::get(shape, *dtype), const_op);
  }
  return const_op.getResult();
}

// Template specialization for double
template <>
std::optional<Value> getConstTensor<double>(PatternRewriter &rewriter,
                                           Operation *op, ArrayRef<double> vec,
                                           ArrayRef<int64_t> shape, std::optional<Type> dtype) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF64Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);

  if (dtype) {
   return rewriter.createOrFold<tosa::CastOp>(
        op->getLoc(), RankedTensorType::get(shape, *dtype), const_op);
  }
  return const_op.getResult();
}

static LogicalResult checkValidityOfCast(Type src, Type dest) {
  if (src == dest)
   return success();

  auto isValid = [](Type ty) {
    return ty.isInteger(1) || ty.isInteger(8) || ty.isInteger(16) ||
           ty.isInteger(32) || ty.isInteger(64) || ty.isBF16() || ty.isF16() || ty.isF32() ||
           ty.isF64();
  };

  return success(isValid(src) && isValid(dest));
}

// Template specialization for float
LogicalResult tosaCastTensorToType(PatternRewriter &rewriter, Operation *op,
                                   Value src, Type destType, Value &result) {

  Type srcElemTy = src.getType().dyn_cast<TensorType>().getElementType();
  Type destElemTy = destType.dyn_cast<TensorType>().getElementType();

  if (failed(checkValidityOfCast(srcElemTy, destElemTy)))
    return rewriter.notifyMatchFailure(
        op, "casting to result dtype is invalid or unsupported");

  if (destElemTy.isInteger(1)) {
    auto srcType = src.getType().dyn_cast<TensorType>();
    SmallVector<int64_t> srcShape(srcType.getShape());
    uint64_t num_total_elements = 1;
    for (int64_t a : srcShape)
      num_total_elements *= a;

    std::optional<Value> constOp;
    if (srcElemTy.isInteger(64)) {
      SmallVector<int64_t> values(num_total_elements, 0);
      constOp =
          tosa::getConstTensor<int64_t>(rewriter, op, values, srcShape).value();
    } else if (srcElemTy.isInteger(32)) {
      SmallVector<int32_t> values(num_total_elements, 0);
      constOp =
          tosa::getConstTensor<int32_t>(rewriter, op, values, srcShape).value();
    } else if (srcElemTy.isInteger(8)) {
      SmallVector<int8_t> values(num_total_elements, 0);
      constOp =
          tosa::getConstTensor<int8_t>(rewriter, op, values, srcShape).value();
    } else if (srcElemTy.isInteger(16)) {
      SmallVector<int16_t> values(num_total_elements, 0);
      constOp =
          tosa::getConstTensor<int16_t>(rewriter, op, values, srcShape).value();
    } else if (srcElemTy.isBF16()) {
      SmallVector<float> values(num_total_elements, 0.0);
      constOp =
          tosa::getConstTensor<float>(rewriter, op, values, srcShape, srcElemTy)
              .value();
    } else if (srcElemTy.isF32()) {
      SmallVector<float> values(num_total_elements, 0.0);
      constOp =
          tosa::getConstTensor<float>(rewriter, op, values, srcShape).value();
    } else if (srcElemTy.isF64()) {
      SmallVector<double> values(num_total_elements, 0.0);
      constOp =
          tosa::getConstTensor<double>(rewriter, op, values, srcShape).value();
    } else {
      op->dump();
      op->emitError("Unsupported conversion to i1");
      return failure();
    }
    Value equalToZero = rewriter.create<tosa::EqualOp>(op->getLoc(), destType,
                                                       src, constOp.value());
    result = rewriter.create<tosa::LogicalNotOp>(op->getLoc(), destType,
                                                 equalToZero);
  } else {
    result = rewriter.create<tosa::CastOp>(op->getLoc(), destType, src);
  }
  return success();
}

Value promoteType(PatternRewriter &rewriter, Value input, TensorType outType) {
  Operation *op = input.getDefiningOp();
  TensorType inType = input.getType().cast<TensorType>();

  if (inType.getElementType() != outType.getElementType()) {
    TensorType promotedType =
        inType.cloneWith(inType.getShape(), outType.getElementType());
    return rewriter.create<tosa::CastOp>(op->getLoc(), promotedType, input);
  }
  return input;
}

TypedValue<RankedTensorType> reshapeTo(Location loc, PatternRewriter &rewriter,
                                       Value val, ArrayRef<int64_t> newShape) {

  auto tensorTy = dyn_cast<TensorType>(val.getType());
  assert(tensorTy);

  auto newTy = RankedTensorType::get(newShape, tensorTy.getElementType());
  return rewriter.create<tosa::ReshapeOp>(
      loc, newTy, val, rewriter.getDenseI64ArrayAttr(newShape));
}

TypedValue<RankedTensorType> transposeBy(Location loc, PatternRewriter &rewriter,
                                        Value val,
                                        ArrayRef<int32_t> permutation) {
  auto tensorTy = dyn_cast<TensorType>(val.getType());
  assert(tensorTy);

  auto permType = RankedTensorType::get({(int64_t)permutation.size()},
                                        rewriter.getI32Type());
  auto permAttr = DenseElementsAttr::get(permType, permutation);
  auto permOp = rewriter.create<tosa::ConstOp>(loc, permType, permAttr);

  SmallVector<int64_t> newShape{tensorTy.getShape()};
  for (size_t i = 0; i < newShape.size(); i++)
    newShape[i] = tensorTy.getShape()[permutation[i]];

  auto newTy = RankedTensorType::get(newShape, tensorTy.getElementType());

  auto v = rewriter.createOrFold<tosa::TransposeOp>(loc, newTy, val, permOp);
  return cast<TypedValue<RankedTensorType>>(v);
}

// Template instantiation
template std::optional<Value> getConstTensor<bool>(PatternRewriter &,
                                                      Operation *,
                                                      ArrayRef<bool> vec,
                                                      ArrayRef<int64_t> shape,
                                                      std::optional<Type> dtype);

template std::optional<Value> getConstTensor<int32_t>(PatternRewriter &,
                                                      Operation *,
                                                      ArrayRef<int32_t> vec,
                                                      ArrayRef<int64_t> shape,
                                                      std::optional<Type> dtype);

template std::optional<Value> getConstTensor<int64_t>(PatternRewriter &,
                                                      Operation *,
                                                      ArrayRef<int64_t> vec,
                                                      ArrayRef<int64_t> shape,
                                                      std::optional<Type> dtype);

LogicalResult getAvgPool2dAccType(PatternRewriter &rewriter, Value input,
                                  TypeAttr &accType) {
  auto inputTy = llvm::dyn_cast<ShapedType>(input.getType());
  if (!inputTy)
    return failure();
  auto inputETy = inputTy.getElementType();

  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(inputETy))
    inputETy = quantType.getStorageType();

  // Tosa supports FP16 and FP32 accumulator type for FP16 input. When the time
  // FP16 is supported, the accumulator type can be selected based on trade-off
  // between performance and accuracy. Set to FP32 by default.
  accType = inputETy.isa<FloatType>()
                ? mlir::TypeAttr::get(rewriter.getF32Type())
                : mlir::TypeAttr::get(rewriter.getIntegerType(32));

  return success();
}

} // namespace tosa
} // namespace mlir
