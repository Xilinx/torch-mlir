//===----------------------------------------------------------------------===//
////
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeCommon.h"
#include "torch-mlir/Conversion/TorchToTosa/TosaLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

// These legalizations are for unary ops with only for floating point datatypes.
// There is no supported quantized integer mode for these.
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenUnaryFPOnlyOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto selfTy = cast<TensorType>(self.getType());

    if (!selfTy)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    if (isa<mlir::FloatType>(selfTy.getElementType())) {
      rewriter.replaceOpWithNewOp<TosaOpT>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          self);
      return success();
    } else {
      return rewriter.notifyMatchFailure(
          op, "Only floating-point datatype legalization supported");
    }
  }
};

// These unary op legalizations are identical for floating-point
// or quantized types
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenUnaryOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TosaOpT>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        adaptor.getSelf());
    return success();
  }
};

// These binary op legalizations are identical for floating-point
// or quantized types
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenBinaryOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsTy = cast<TensorType>(lhs.getType());
    Value rhs = adaptor.getOther();
    auto rhsTy = cast<TensorType>(rhs.getType());

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    auto outTy = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    auto binaryOp =
        tosa::createBinaryOpAndCast<TosaOpT>(rewriter, op, outTy, lhs, rhs);
    rewriter.replaceOp(op, binaryOp.getResult());
    return success();
  }
};

template <typename T>
static bool isInValidRange(bool isFloat, const double &doubleValue, bool isInt,
                           const int64_t &intValue) {
  if (isFloat) {
    // Do a round-trip check here instead of numeric limits due to
    // compiler warnings around double <-> int conversion.
    return (doubleValue == static_cast<double>(static_cast<T>(doubleValue)));
  } else {
    assert(isInt);
    return (intValue >= static_cast<int64_t>(std::numeric_limits<T>::min())) &&
           (intValue <= static_cast<int64_t>(std::numeric_limits<T>::max()));
  }
  return true;
}

// FIXME: This will eventually go into a Tosa*Utils file.
LogicalResult torchScalarToTosaTensor(ConversionPatternRewriter &rewriter,
                                      Operation *op, Value torchScalarValue,
                                      Value &tosaTensor, Type dtype,
                                      llvm::ArrayRef<int64_t> dshape) {
  // Retrieve a const float or int value but create the out Tensor with dtype.
  double doubleValue;
  auto isFloat =
      matchPattern(torchScalarValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(torchScalarValue, m_TorchConstantInt(&intValue));

  if (!isFloat && !isInt)
    return rewriter.notifyMatchFailure(op,
                                       "Unable to extract the scalar constant");

  if (isa<mlir::FloatType>(dtype)) {
    tosaTensor = tosa::getConstTensor<float>(rewriter, op,
                                             (isFloat ? doubleValue : intValue),
                                             dshape, dtype)
                     .value();
  } else if (auto intType = dyn_cast<mlir::IntegerType>(dtype)) {
    auto w = intType.getWidth();
    if (w != 1 && w != 32 && w != 64)
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "Unsupported integer type: " << intType;
      });

    if (w == 1) {
      if (!isInValidRange<bool>(isFloat, doubleValue, isInt, intValue)) {
        return rewriter.notifyMatchFailure(
            op, "Supplied value of scalar constant exceeds limits "
                "of destination type");
      }
      bool d = isFloat ? static_cast<bool>(doubleValue)
                       : static_cast<bool>(intValue);
      tosaTensor =
          tosa::getConstTensor<bool>(rewriter, op, {d}, dshape).value();
    } else if (w == 32) {
      if (!isInValidRange<int32_t>(isFloat, doubleValue, isInt, intValue)) {
        return rewriter.notifyMatchFailure(
            op, "Supplied value of scalar constant exceeds limits "
                "of destination type");
      }
      int32_t d = isFloat ? static_cast<int32_t>(doubleValue)
                          : static_cast<int32_t>(intValue);
      tosaTensor =
          tosa::getConstTensor<int32_t>(rewriter, op, {d}, dshape).value();
    } else if (w == 64) {
      if (!isInValidRange<int64_t>(isFloat, doubleValue, isInt, intValue)) {
        return rewriter.notifyMatchFailure(
            op, "Supplied value of scalar constant exceeds limits "
                "of destination type");
      }
      int64_t d = (isFloat ? static_cast<int64_t>(doubleValue) : intValue);
      tosaTensor =
          tosa::getConstTensor<int64_t>(rewriter, op, {d}, dshape).value();
    }
  } else {
    return rewriter.notifyMatchFailure(op, "Usupported element type");
  }

  return success();
}

LogicalResult torchAlphaToTosaTensor(ConversionPatternRewriter &rewriter,
                                     Operation *op, Value alphaScalar,
                                     Value &alphaTensor, Type dtype,
                                     bool checkForUnity) {
  if (succeeded(torchScalarToTosaTensor(rewriter, op, alphaScalar, alphaTensor,
                                        dtype, {})))
    return success();

  // `alpha` has not been specified.
  int64_t alphaValue;
  if (!matchPattern(alphaScalar, m_TorchConstantInt(&alphaValue)))
    return rewriter.notifyMatchFailure(
        op, "Currently only scalar constants are supported for "
            "alpha in TOSA operation");
  // When no alpha has been specified, this must be 1.
  if (checkForUnity && alphaValue != 1)
    return rewriter.notifyMatchFailure(op,
                                       "Unsupported integer value for alpha");

  alphaTensor = tosa::getConstTensor<float>(
                    rewriter, op, {static_cast<float>(alphaValue)}, {}, dtype)
                    .value();

  return success();
}

// These binary op legalizations are specific to add/sub which have an
// alpha multiplier.
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenAddSubOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // left  : tensor: tensor<i32/i64/f32>
    // right : scalar: i32/i64/f32
    //         tensor: tensor<i32/i64/f32>
    // alpha : scalar: i32/i64/f32
    // output: tensor: tensor<i32/i64/f32>
    Value lhs = adaptor.getSelf();
    auto lhsType = dyn_cast<TensorType>(lhs.getType());
    Value rhs = adaptor.getOther();
    auto rhsType = dyn_cast<TensorType>(rhs.getType());

    if (!lhsType)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    if (auto lhsElemTy = dyn_cast<IntegerType>(lhsType.getElementType())) {
      if (lhsElemTy.getWidth() > 64)
        return rewriter.notifyMatchFailure(
            op, "Integers with widths greater than 64 are not supported");
    }

    // Get output type: tensor<i32/i64/f32>
    auto outType = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");
    }

    if (!rhsType) {
      if (failed(torchScalarToTosaTensor(rewriter, op, op.getOther(), rhs,
                                         outElemTy, {}))) {
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA operation");
      }
      rhsType = dyn_cast<TensorType>(rhs.getType());
    }

    // aten.rsub(lhs, rhs, alpha) computes rhs - lhs * alpha
    if constexpr (std::is_same<AtenOpT, AtenRsubScalarOp>::value) {
      std::swap(lhs, rhs);
      std::swap(lhsType, rhsType);
    }

    Type rhsAlphaMulElemType;
    if (isa<mlir::FloatType>(outElemTy)) {
      rhsAlphaMulElemType = outElemTy;
    } else {
      // if output type is 64, input type should also be 32
      rhsAlphaMulElemType = rewriter.getIntegerType(32);
    }

    if (rhsType.getElementType() != rhsAlphaMulElemType) {
      // right is tensor, rhsType == tensor<i32/i64/f32>
      // right must be cast to same type as the alpha, so MulOp success
      rhsType = RankedTensorType::get(rhsType.getShape(), rhsAlphaMulElemType);
      rhs = rewriter.create<tosa::CastOp>(op->getLoc(), rhsType, rhs);
    }

    // Handle scalar value alpha.
    // It should be either f32/i32
    Value alphaTensor;
    if (failed(torchAlphaToTosaTensor(rewriter, op.getOperation(),
                                      op.getAlpha(), alphaTensor,
                                      rhsAlphaMulElemType,
                                      /*checkForUnity=*/false))) {
      return rewriter.notifyMatchFailure(
          op, "Currently only scalar constants are supported for "
              "alpha in conversion to TOSA operation");
    }

    auto mulAlphaOp = tosa::createMulOpAndCast(rewriter, op, rhsType, rhs,
                                               alphaTensor, /*shift=*/0);

    if (outElemTy.isInteger(64)) {
      // Tosa doesn't support 64-bit elementwise addition and subtraction.
      // if outElemTy tensor<i64>, mulTensor must be tensor<i32>,
      //    left value could be tensor<f32/i32/i64> type, cast left value to
      //    tensor<i32> type
      auto addOrSubi64Op = tosa::createBinaryOpAndCast<TosaOpT>(
          rewriter, op,
          RankedTensorType::get(outType.getShape(), rhsAlphaMulElemType), lhs,
          mulAlphaOp);

      // cast tensor<i32> back to tensor<i64>
      rewriter.replaceOpWithNewOp<tosa::CastOp>(op, outType, addOrSubi64Op);
      return success();
    }

    auto binaryOp = tosa::createBinaryOpAndCast<TosaOpT>(rewriter, op, outType,
                                                         lhs, mulAlphaOp);
    rewriter.replaceOp(op, binaryOp.getResult());
    return success();
  }
}; // namespace

// Binary op legalizations for comparator ops.
template <typename AtenOpT, typename TosaOpT>
class ConvertAtenCompareOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsTy = dyn_cast<TensorType>(lhs.getType());
    Value rhs = adaptor.getOther();
    auto rhsTy = dyn_cast<TensorType>(rhs.getType());

    if (!lhsTy)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    auto lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    // For bitwise operators, only integer datatype legalization is supported
    constexpr bool isBitwiseOp =
        std::is_same<AtenOpT, AtenBitwiseAndTensorOp>() ||
        std::is_same<AtenOpT, AtenBitwiseOrTensorOp>() ||
        std::is_same<AtenOpT, AtenBitwiseXorTensorOp>();
    if (isa<mlir::FloatType>(lhsElemTy) && isBitwiseOp) {
      return rewriter.notifyMatchFailure(op,
                                         "For bitwise operators, only integer "
                                         "datatype legalization is supported");
    }

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(torchScalarToTosaTensor(rewriter, op, op.getOther(),
                                         rhsAsTensor, lhsElemTy, {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA operation");
    }
    auto rhsTensor = rhsTy ? rhs : rhsAsTensor;
    // There is no Lesser operator in TOSA.
    constexpr auto swapLhsRhs = (std::is_same<AtenOpT, AtenLtTensorOp>() ||
                                 std::is_same<AtenOpT, AtenLtScalarOp>() ||
                                 std::is_same<AtenOpT, AtenLeTensorOp>() ||
                                 std::is_same<AtenOpT, AtenLeScalarOp>() ||
                                 std::is_same<AtenOpT, AtenLeTensorOp>());

    // Promote lhs and rhs dtypes for bitwise operators.
    TensorType resultTy = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    if (isBitwiseOp) {
      lhs = tosa::promoteType(rewriter, lhs, resultTy);
      rhsTensor = tosa::promoteType(rewriter, rhsTensor, resultTy);
    }

    auto resultOp = rewriter.create<TosaOpT>(op.getLoc(), resultTy,
                                             (swapLhsRhs ? rhsTensor : lhs),
                                             (swapLhsRhs ? lhs : rhsTensor));

    // There is no NE operator in TOSA.
    if constexpr (std::is_same<AtenOpT, AtenNeTensorOp>() ||
                  std::is_same<AtenOpT, AtenNeScalarOp>()) {
      rewriter.replaceOpWithNewOp<tosa::LogicalNotOp>(op, resultTy,
                                                      resultOp.getResult());
    }

    else {
      rewriter.replaceOp(op, resultOp.getResult());
    }

    return success();
  }
};

// Binary op legalizations for Mul variants.
template <typename AtenOpT>
class ConvertAtenMulOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsType = dyn_cast<TensorType>(lhs.getType());

    if (!lhsType)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    auto outType = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    Value rhsTensor;
    if constexpr (std::is_same<AtenOpT, AtenSquareOp>()) {
      rhsTensor = lhs;
    } else {
      Value rhsAsTensor;
      Value rhs = adaptor.getOther();
      auto rhsType = dyn_cast<TensorType>(rhs.getType());
      if (!rhsType) {
        if (failed(torchScalarToTosaTensor(rewriter, op, op.getOther(),
                                           rhsAsTensor, outElemTy, {}))) {
          return rewriter.notifyMatchFailure(
              op, "Currently only scalar constants are supported for "
                  "conversion in TOSA operation");
        }
      }
      rhsTensor = rhsType ? rhs : rhsAsTensor;
    }

    if (isa<mlir::FloatType>(outElemTy) || isa<mlir::IntegerType>(outElemTy)) {
      auto outType = cast<TensorType>(
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()));

      auto mulOp = tosa::createMulOpAndCast(rewriter, op, outType, lhs,
                                            rhsTensor, /*shift=*/0);
      rewriter.replaceOp(op, mulOp.getResult());
      return success();
    }

    // Quantized multiplication may need to rescale inputs.
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype "
            "legalization currently supported");
  }
};

template <typename AtenOpT>
class ConvertAtenDivOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsTy = dyn_cast<TensorType>(lhs.getType());
    Value rhs = adaptor.getOther();
    auto rhsTy = dyn_cast<TensorType>(rhs.getType());

    if (!lhsTy)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    auto lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(torchScalarToTosaTensor(rewriter, op, op.getOther(),
                                         rhsAsTensor, lhsElemTy, {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA operation");
    }
    auto rhsTensor = rhsTy ? rhs : rhsAsTensor;
    auto outType = cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    // auto result;
    Value result;
    if (isa<mlir::FloatType>(outType.getElementType())) {
      // The input to the reciprocal is an integer sometimes, and we may need to
      // promote it to a floating point. Per TOSA specification, the input types
      // can only be floating point for tosa::ReciprocalOp.
      Value rhsCasted = tosa::promoteType(rewriter, rhsTensor, outType);
      auto rcpOp = rewriter.create<tosa::ReciprocalOp>(
          op->getLoc(), rhsCasted.getType(), rhsCasted);

      result = tosa::createMulOpAndCast(rewriter, op, outType, lhs,
                                        rcpOp.getResult(), /*shift=*/0)
                   .getResult();
    } else {
      // The output type can be different than the input types (e.g. dividing an
      // int tensor results in a floating point tensor).
      result = tosa::createBinaryOpAndCast<tosa::IntDivOp>(
                   rewriter, op, outType, lhs, rhsTensor)
                   .getResult();
    }

    rewriter.replaceOp(op, {result});
    return success();
  }
};

// This defines a template to construct ops whose legalizations are
// specialized.
template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <>
LogicalResult ConvertAtenOp<AtenTanhOp>::matchAndRewrite(
    AtenTanhOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  auto selfTy = cast<TensorType>(self.getType());
  if (selfTy && isa<mlir::FloatType>(selfTy.getElementType())) {
    rewriter.replaceOpWithNewOp<tosa::TanhOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  }
  // Sigmoid legalization in TOSA for quantized element-type uses specialized
  // tosa.table construct.
  return rewriter.notifyMatchFailure(
      op, "Only floating-point datatype legalization currently supported");
}

template <>
LogicalResult ConvertAtenOp<AtenSigmoidOp>::matchAndRewrite(
    AtenSigmoidOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  auto selfTy = cast<TensorType>(self.getType());
  if (selfTy && isa<mlir::FloatType>(selfTy.getElementType())) {
    rewriter.replaceOpWithNewOp<tosa::SigmoidOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  }
  // Sigmoid legalization in TOSA for quantized element-type uses
  // specialized tosa.table construct.
  return rewriter.notifyMatchFailure(
      op, "Only floating-point datatype legalization currently supported");
}

template <>
LogicalResult ConvertAtenOp<AtenReluOp>::matchAndRewrite(
    AtenReluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value self = adaptor.getSelf();
  auto selfTy = cast<TensorType>(self.getType());

  // Maps to tosa.clamp which has both int and fp limits.
  int64_t clampMin = 0;
  Value clampIn = self;
  if (!selfTy) {
    return rewriter.notifyMatchFailure(op,
                                       "Only Tensor types supported in TOSA");
  }

  // Rescale the clampIn for quantized types. TBD
  if (!isa<mlir::FloatType>(selfTy.getElementType())) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization currently supported");
  }
  rewriter.replaceOpWithNewOp<tosa::ClampOp>(
      op, getTypeConverter()->convertType(op.getType()), clampIn,
      rewriter.getI64IntegerAttr(clampMin),
      rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max()),
      rewriter.getF32FloatAttr(0.0f),
      rewriter.getF32FloatAttr(std::numeric_limits<float>::max()));
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenLeakyReluOp>::matchAndRewrite(
    AtenLeakyReluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value self = adaptor.getSelf();
  auto selfTy = cast<TensorType>(self.getType());
  if (!isa<mlir::FloatType>(selfTy.getElementType())) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization currently supported");
  }

  Value alphaScalar = op.getNegativeSlope();
  Value alphaTensor;
  if (failed(torchScalarToTosaTensor(rewriter, op.getOperation(), alphaScalar,
                                     alphaTensor, selfTy.getElementType(), {})))
    return rewriter.notifyMatchFailure(
        op, "Negative slope needs to be a scalar constant for conversion to "
            "TOSA LeakyReLU operation");

  auto zero =
      tosa::getConstTensor<float>(rewriter, op, 0, {}, selfTy.getElementType())
          .value();
  auto cond = rewriter.create<tosa::GreaterEqualOp>(
      op->getLoc(),
      RankedTensorType::get(selfTy.getShape(), rewriter.getIntegerType(1)),
      self, zero);
  auto mulTensor = rewriter.create<tosa::MulOp>(
      op->getLoc(), getTypeConverter()->convertType(op.getType()), self,
      alphaTensor, /*shift=*/0);

  rewriter.replaceOpWithNewOp<tosa::SelectOp>(
      op, getTypeConverter()->convertType(op.getType()), cond, self, mulTensor);

  return success();
}

using ReductionConvFunc = std::optional<Value> (*)(PatternRewriter &,
                                                   Operation *,
                                                   RankedTensorType, Value,
                                                   ElementsAttr, bool);

// They all constitute a common form invoking the appropriate
// converion function in TosaLegalizeCommon.cpp
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenReductionOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  // Each variant must implement corresponding parameter parsing options
  virtual LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
      ElementsAttr &reduceDimsAttr, bool &keepDims) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented reduce_dims and keep_dims parsing function");
  }

  // Common rewriter for all reduction ops, calls the specific implementation of
  // readReduceDimsAndKeepDims() needed for the op variant.
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto selfTy = cast<TensorType>(self.getType());

    if (!selfTy)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    auto outputTy = cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    if (!outputTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor type outputs permitted for reduce_mean");

    ElementsAttr reduceDimsAttr;
    bool keepDims;

    if (failed(readReduceDimsAndKeepDims(op, adaptor, rewriter, reduceDimsAttr,
                                         keepDims)))
      return failure();

    std::optional<Value> result =
        ConversionFuncT(rewriter, op, outputTy, self, reduceDimsAttr, keepDims);

    if (!result)
      return failure();

    // TBD - support dtype casting.

    rewriter.replaceOp(op, {result.value()});

    return success();
  }
};

// This reduction op legalization template handles op variants that have
// explicit reduce_dims dimensions (provided as a list) and keep_dims
// parameters.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenMultipleDimsReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
  using ConvertAtenReductionOp<AtenOpT,
                               ConversionFuncT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(AtenOpT op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          ElementsAttr &reduceDimsAttr,
                                          bool &keepDims) const override {
    SmallVector<int64_t, 4> reduceDims;
    if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(reduceDims)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");
    int64_t N = reduceDims.size();
    int64_t inputRank =
        cast<RankedTensorType>(adaptor.getSelf().getType()).getRank();
    for (unsigned i = 0; i < N; i++) {
      reduceDims[i] = toPositiveDim(reduceDims[i], inputRank);
      if (!isValidDim(reduceDims[i], inputRank))
        return rewriter.notifyMatchFailure(op,
                                           "reduce dim is statically invalid");
    }
    auto reduceDimsType = RankedTensorType::get({N}, rewriter.getI64Type());
    reduceDimsAttr =
        DenseIntElementsAttr::get(reduceDimsType, llvm::ArrayRef(reduceDims));

    keepDims = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDims)))
      return rewriter.notifyMatchFailure(
          op, "non-const keepdim parameter unsupported");

    return success();
  }
};

// This reduction op legalization template handles op variants that reduce in
// only one explicit dim which is provided as a number (rather than a list), and
// a keep_dims parameter.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenOneDimReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
  using ConvertAtenReductionOp<AtenOpT,
                               ConversionFuncT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(AtenOpT op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          ElementsAttr &reduceDimsAttr,
                                          bool &keepDims) const override {
    int64_t reduceDim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&reduceDim)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");
    int64_t inputRank =
        cast<RankedTensorType>(adaptor.getSelf().getType()).getRank();
    reduceDim = toPositiveDim(reduceDim, inputRank);
    if (!isValidDim(reduceDim, inputRank))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");
    auto reduceDimsType = RankedTensorType::get({1}, rewriter.getI64Type());
    reduceDimsAttr =
        DenseIntElementsAttr::get(reduceDimsType, llvm::ArrayRef({reduceDim}));

    keepDims = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDims)))
      return rewriter.notifyMatchFailure(
          op, "non-const keepdim parameter unsupported");

    return success();
  }
};

// This reduction op legalization template handles op variants that reduce all
// dims does not keep dims.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenAllDimsReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
public:
  using ConvertAtenReductionOp<AtenOpT,
                               ConversionFuncT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(AtenOpT op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          ElementsAttr &reduceDimsAttr,
                                          bool &keepDims) const override {
    auto self = adaptor.getSelf();
    auto selfTy = cast<RankedTensorType>(self.getType());

    // Select all dims to reduce
    SmallVector<int64_t, 4> reduceDims;
    for (int64_t i = 0; i < selfTy.getRank(); i++)
      reduceDims.push_back(i);
    int64_t N = selfTy.getRank();
    auto reduceDimsType = RankedTensorType::get({N}, rewriter.getI64Type());
    reduceDimsAttr =
        DenseIntElementsAttr::get(reduceDimsType, llvm::ArrayRef(reduceDims));
    keepDims = false;

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenArgmaxOp>::matchAndRewrite(
    AtenArgmaxOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());

  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in TOSA argmax");

  int64_t reduceDim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&reduceDim))) {
    // NoneType indicates reduce on all dims
    reduceDim = -1;
  } else {
    int64_t inputRank = selfTy.getRank();
    reduceDim = toPositiveDim(reduceDim, inputRank);
    if (!isValidDim(reduceDim, inputRank))
      return rewriter.notifyMatchFailure(op,
                                         "reduce dim is statically invalid");
  }

  bool keepDim = false;
  if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
    return rewriter.notifyMatchFailure(
        op, "non-const keepdim parameter unsupported");

  auto resultTy = cast<RankedTensorType>(
      getTypeConverter()->convertType(op.getResult().getType()));
  auto outputETy = resultTy.getElementType();

  // Create a single instance of tosa.argmax.
  // Multiple dims require chained construct.
  auto buildArgmax = [&](int64_t reduceDim, Value input) -> Value {
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
    SmallVector<int64_t> outputShapeArr = {};
    int32_t i = 0;

    for (auto &dim : inputShape) {
      if (i++ != reduceDim) {
        outputShapeArr.push_back(dim);
      } else {
        if (keepDim)
          outputShapeArr.push_back(1);
      }
    }

    // Tosa argmax output is i32, while Torch backend mandates i64.
    auto outputReduceTy = RankedTensorType::get(
        makeShapeLLVMCompatible(ArrayRef<int64_t>(outputShapeArr)),
        rewriter.getI32Type());
    auto reduceDimAttr =
        rewriter.getIntegerAttr(rewriter.getI64Type(), reduceDim);
    return rewriter
        .create<tosa::ArgMaxOp>(op->getLoc(),
                                getTypeConverter()->convertType(outputReduceTy),
                                input, reduceDimAttr)
        .getResult();
  };

  // Convert the final index to i64 for backend finalization, However, i64
  // is not a defined type for tosa.cast, so using arith.extsi instead.
  auto castToInt64 = [&](Value result) -> LogicalResult {
    auto resTy = cast<ShapedType>(result.getType());
    if (!resTy)
      return rewriter.notifyMatchFailure(op,
                                         "Argmax: Result is not a shaped type");

    auto resShape = makeShapeTorchCompatible(resTy.getShape());
    auto outTy =
        RankedTensorType::get(makeShapeLLVMCompatible(resShape), outputETy);

    rewriter.replaceOpWithNewOp<arith::ExtSIOp>(
        op, getTypeConverter()->convertType(outTy), result);

    return success();
  };

  if (reduceDim == -1) { // reducing on all dims
    Value input = self;
    for (int dim = 0; dim < selfTy.getRank(); dim++) {
      // progressively reduce each 0-th dim
      input = buildArgmax(0, input);
    }
    return castToInt64(input);
  } else {
    return castToInt64(buildArgmax(reduceDim, self));
  }

  return success();
}

template <typename AtenOpT>
class ConvertAtenSqueezeOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  // Each variant must implement corresponding parameter parsing options
  virtual LogicalResult
  generateSqueezedShape(AtenOpT op, RankedTensorType selfTy,
                        ConversionPatternRewriter &rewriter,
                        SmallVector<int64_t> &squeezedShape) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented dim/dim-list parsing function");
  }

  // Common rewriter for all squeeze ops, calls the specific implementation of
  // generateSqueezedShape() needed for the op variant.
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto selfTy = cast<RankedTensorType>(self.getType());

    if (!selfTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA argmax");

    SmallVector<int64_t> newOutputShape;
    if (failed(generateSqueezedShape(op, selfTy, rewriter, newOutputShape)))
      return rewriter.notifyMatchFailure(op,
                                         "Squeeze could not compute new shape");

    auto resultTy = cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getResult().getType()));
    auto resultElemTy = resultTy.getElementType();

    auto newOutputTy = RankedTensorType::get(
        makeShapeLLVMCompatible(newOutputShape), resultElemTy);

    auto reshapeOp = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(),
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            newOutputTy),
        self, rewriter.getDenseI64ArrayAttr(newOutputShape));
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            newOutputTy),
        reshapeOp);

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenSqueezeOneDimOp : public ConvertAtenSqueezeOp<AtenOpT> {
  using ConvertAtenSqueezeOp<AtenOpT>::ConvertAtenSqueezeOp;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  generateSqueezedShape(AtenOpT op, RankedTensorType selfTy,
                        ConversionPatternRewriter &rewriter,
                        SmallVector<int64_t> &squeezedShape) const override {
    int64_t squeezeDim;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&squeezeDim)))
      return rewriter.notifyMatchFailure(op,
                                         "non-const dim parameter unsupported");

    // Handle negative dim
    if (squeezeDim < 0)
      squeezeDim = squeezeDim + selfTy.getRank();

    auto selfShape = makeShapeTorchCompatible(selfTy.getShape());

    // Only dims statically known to have size=1 are reduced.
    // Dynamic dims are treated as unknowns and will not be squeezed
    // even if dim parameter says it should be.
    uint32_t dimNum = 0;
    for (auto &dim : selfShape) {
      if (dim != 1 || squeezeDim != dimNum)
        squeezedShape.push_back(dim);
      dimNum++;
    }

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenSqueezeAllDimsOp : public ConvertAtenSqueezeOp<AtenOpT> {
  using ConvertAtenSqueezeOp<AtenOpT>::ConvertAtenSqueezeOp;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  generateSqueezedShape(AtenOpT op, RankedTensorType selfTy,
                        ConversionPatternRewriter &rewriter,
                        SmallVector<int64_t> &squeezedShape) const override {
    auto selfShape = makeShapeTorchCompatible(selfTy.getShape());

    // Dims that may dynamically resolve to 1 are not reduced here. Only
    // compile-time resolvable dims are handled here.
    for (auto &dim : selfShape) {
      if (dim != 1)
        squeezedShape.push_back(dim);
    }
    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenPowScalarOp>::matchAndRewrite(
    AtenPowScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value exp = adaptor.getExponent();
  auto expTy = dyn_cast<RankedTensorType>(exp.getType());

  if (!expTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in TOSA Pow");

  if (!isa<mlir::FloatType>(expTy.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization supported");

  Value selfTensor;
  Value selfScalar = op.getSelf();
  if (failed(torchScalarToTosaTensor(rewriter, op, selfScalar, selfTensor,
                                     expTy.getElementType(), {})))
    return rewriter.notifyMatchFailure(
        op, "Currently only scalar constants are supported for "
            "conversion in TOSA Pow operation");

  auto outType =
      cast<TensorType>(getTypeConverter()->convertType(op.getType()));

  auto powOp = tosa::createBinaryOpAndCast<tosa::PowOp>(rewriter, op, outType,
                                                        selfTensor, exp);
  rewriter.replaceOp(op, powOp.getResult());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenPowTensorScalarOp>::matchAndRewrite(
    AtenPowTensorScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());

  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in TOSA Pow");

  if (!isa<mlir::FloatType>(selfTy.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization supported");

  auto outType =
      cast<TensorType>(getTypeConverter()->convertType(op.getType()));

  Value expTensor;
  Value expScalar = op.getExponent();
  if (failed(torchScalarToTosaTensor(rewriter, op, expScalar, expTensor,
                                     outType.getElementType(), {})))
    return rewriter.notifyMatchFailure(
        op, "Currently only scalar constants are supported for "
            "conversion in TOSA Pow operation");

  auto powOp = tosa::createBinaryOpAndCast<tosa::PowOp>(rewriter, op, outType,
                                                        self, expTensor);
  rewriter.replaceOp(op, powOp.getResult());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenPowTensorTensorOp>::matchAndRewrite(
    AtenPowTensorTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());

  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in TOSA Pow");

  if (!isa<mlir::FloatType>(selfTy.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization supported");

  auto outType =
      cast<TensorType>(getTypeConverter()->convertType(op.getType()));

  Value expTensor = adaptor.getExponent();
  if (expTensor.getType() != selfTy) {
    expTensor = rewriter.createOrFold<tosa::CastOp>(
        op->getLoc(),
        RankedTensorType::get(outType.getShape(), selfTy.getElementType()),
        expTensor);
  }

  auto powOp = tosa::createBinaryOpAndCast<tosa::PowOp>(rewriter, op, outType,
                                                        self, expTensor);
  rewriter.replaceOp(op, powOp.getResult());
  return success();
}

Type getMatMulOutputType(Type inputElemTy, Type outputElemTy,
                         PatternRewriter &rewriter) {
  Type tosaOutputElemTy;
  if (auto floatTy = dyn_cast<mlir::FloatType>(inputElemTy)) {
    if (inputElemTy.isF16() && outputElemTy.isF16()) {
      return rewriter.getF16Type();
    }
    if (floatTy.isBF16() || floatTy.isF16() || floatTy.isF32()) {
      // Always accumulate on f32
      tosaOutputElemTy = rewriter.getF32Type();
    }
  } else if (auto integerTy = dyn_cast<IntegerType>(inputElemTy)) {
    if (integerTy.isInteger(/*width=*/8)) {
      tosaOutputElemTy = rewriter.getIntegerType(/*width=*/32);
    } else if (integerTy.isInteger(/*width=*/16)) {
      tosaOutputElemTy = rewriter.getIntegerType(/*width=*/48);
    }
  }
  return tosaOutputElemTy;
}

RankedTensorType getCastedInputTypeForMatmul(Value inputValue,
                                             PatternRewriter &rewriter) {
  // Check to see if the inputs to the matmul where casted from another type
  auto preCastType =
      TypeSwitch<Operation *, RankedTensorType>(inputValue.getDefiningOp())
          .Case([](tosa::CastOp op) {
            return cast<RankedTensorType>(op->getOperand(0).getType());
          })
          .Default([](Operation * /*op*/) { return RankedTensorType(); });
  if (!preCastType) {
    return preCastType;
  }
  Type castOutputTy =
      cast<RankedTensorType>(inputValue.getType()).getElementType();
  // The FxImporter does not support si48 and neither does torch-mlir so for now
  // we reject this case for the future when the dialect and importer may
  // support it.
  if (castOutputTy.isInteger(48) &&
      (castOutputTy.isSignedInteger() || castOutputTy.isSignlessInteger())) {
    return RankedTensorType();
  }
  // Calculate the expected accumulator type based on the input type of the cast
  auto accumulatorType =
      getMatMulOutputType(preCastType.getElementType(), castOutputTy, rewriter);
  // If the expected accumulatorType for the given input type of the
  // cast matches the output type of the cast then we can fold the
  // casting into the matmul. The tosa matmul is defined to cast the
  // inputs to the output type first, so we do not need explicit
  // casts up front.
  return accumulatorType == castOutputTy ? preCastType : RankedTensorType();
}

// Perform the basic n-dim matmul operation encompassing the handling of
// broadcasting and dynamic shape propagation.
// All PyTorch ops that leverage matrix multiplication will derive this and
// implement their specialized input processing (e.g transpose), and output
// processing, e.g. GEMM or fully connected bias handling.
template <typename AtenOpT>
class ConvertAtenMatmulBaseOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  // Each variant must implement corresponding parameter parsing options.
  // Maintain separate input read functions for each variant because it is not
  // necessarily true with all variants that the first two operands are the lhs
  // and rhs.
  virtual LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                         ConversionPatternRewriter &rewriter,
                                         Value &lhs, Value &rhs) const {
    return rewriter.notifyMatchFailure(
        op,
        "Unimplemented matrix multiplication variant input parsing function");
  }
  LogicalResult performMatmul(AtenOpT op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &lhs,
                              Value &rhs, Value &output) const {

    auto lhsTy = cast<RankedTensorType>(lhs.getType());
    auto rhsTy = cast<RankedTensorType>(rhs.getType());

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    auto lhsShape = makeShapeTorchCompatible(lhsTy.getShape());
    auto rhsShape = makeShapeTorchCompatible(rhsTy.getShape());

    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy)
      return rewriter.notifyMatchFailure(op,
                                         "Matmul: input datatypes mismatched");

    // Step: check if the inputs have been casted from a supported input type to
    // an accumulator type and insert casts back to the original type if true
    RankedTensorType lhsPreCastedType =
        getCastedInputTypeForMatmul(lhs, rewriter);
    RankedTensorType rhsPreCastedType =
        getCastedInputTypeForMatmul(rhs, rewriter);
    if (lhsPreCastedType && rhsPreCastedType &&
        (lhsPreCastedType.getElementType() ==
         rhsPreCastedType.getElementType())) {
      lhs = rewriter.create<tosa::CastOp>(
          lhs.getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              lhsPreCastedType),
          lhs);
      rhs = rewriter.create<tosa::CastOp>(
          rhs.getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              rhsPreCastedType),
          rhs);
      lhsElemTy = cast<RankedTensorType>(lhsPreCastedType).getElementType();
      rhsElemTy = cast<RankedTensorType>(rhsPreCastedType).getElementType();
    }

    auto torchMatmulOutputType =
        cast<torch::Torch::ValueTensorType>(op.getType()).getDtype();
    auto outputElemTy =
        getMatMulOutputType(lhsElemTy, torchMatmulOutputType, rewriter);
    if (!outputElemTy) {
      return rewriter.notifyMatchFailure(
          op, "Only i8 and i16 integer and bf16, f16 and "
              "f32 float types are valid");
    }

    // Legalization constructs may offer input shapes but expect output shapes
    // to be inferred, e.g.
    // func @forward(%arg0: !torch.vtensor<[14,19],f32>,
    //               %arg1: !torch.vtensor<[19,28],f32>) ->
    //               !torch.vtensor<[?,?],f32>
    // This is tricky with matmul, since TOSA matmul is on 3D inputs.
    // This means the need to reshape potentially both inputs and outputs,
    // and reshape to unknown shape is undefined.

    auto maxInputRank = lhsRank > rhsRank ? lhsRank : rhsRank;
    // If performing dot product on vectors, the RHS is synthetically transposed
    if (maxInputRank == 1)
      maxInputRank++;

    // Obtaining the rank broadcasted shapes of tensors makes it easier to
    // construct the input and output reshaping logic.
    auto getRankBroadcastedShape = [&](Value tensor,
                                       bool isRHS) -> SmallVector<int64_t> {
      auto tensorTy = cast<TensorType>(tensor.getType());
      auto tensorShape = makeShapeTorchCompatible(tensorTy.getShape());
      auto tensorRank = tensorTy.getRank();

      SmallVector<int64_t> bcastedShape;

      auto bcastDims = maxInputRank - tensorRank;

      if (isRHS && (tensorRank == 1) && bcastDims) {
        // RHS with rank1 is special. It be synthetically transposed to dim[:-2]
        for (int32_t i = 0; i < bcastDims - 1; i++)
          bcastedShape.push_back(1);
        bcastedShape.push_back(tensorShape[0]);
        bcastedShape.push_back(1);
      } else {
        if (bcastDims > 0) { // rank broadcast
          for (uint32_t i = 0; i < bcastDims; i++)
            bcastedShape.push_back(1);
        }
        for (auto &dim : tensorShape)
          bcastedShape.push_back(dim);
      }
      return bcastedShape;
    };

    // Step: Rank broadcast the two inputs.
    auto lhsBroadcastedShape = getRankBroadcastedShape(lhs, false);
    auto lhsBroadcastedTy = RankedTensorType::get(
        makeShapeLLVMCompatible(lhsBroadcastedShape), lhsElemTy);
    auto rhsBroadcastedShape = getRankBroadcastedShape(rhs, true);
    auto rhsBroadcastedTy = RankedTensorType::get(
        makeShapeLLVMCompatible(rhsBroadcastedShape), rhsElemTy);

    auto rankBroadcastedLhs =
        lhsRank == maxInputRank
            ? lhs
            : rewriter.create<tosa::ReshapeOp>(
                  op->getLoc(),
                  OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                      lhsBroadcastedTy),
                  lhs, rewriter.getDenseI64ArrayAttr(lhsBroadcastedShape));

    auto rankBroadcastedRhs =
        rhsRank == maxInputRank
            ? rhs
            : rewriter.create<tosa::ReshapeOp>(
                  op->getLoc(),
                  OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                      rhsBroadcastedTy),
                  rhs, rewriter.getDenseI64ArrayAttr(rhsBroadcastedShape));

    // TOSA matmul is performed on two 3D inputs and generates a 3D output.
    // Lower ranked tensors are dim-1 reshaped up to 3D
    auto reshapeUpTo3DTensor = [&](Value tensor) -> Value {
      auto tensorTy = cast<TensorType>(tensor.getType());
      auto rank = tensorTy.getRank();

      assert(rank <= 3 && "reshapeUpTo3D tensor must receive rank <= 3");
      if (rank == 3)
        return tensor;

      auto shape = makeShapeTorchCompatible(tensorTy.getShape());
      SmallVector<int64_t> newShape({1, 1, 1});

      if (rank == 2) { // batchsize = 1
        newShape[1] = shape[0];
        newShape[2] = shape[1];
      } else { // rank 1
        newShape[2] = shape[0];
      }
      auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                           tensorTy.getElementType());

      return rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newType),
          tensor, rewriter.getDenseI64ArrayAttr(newShape));
    };

    // Where broadcasting is required in one or more batch dims, the following
    // is done.
    // Where all batch dims are involved in broadcasting:
    // Given A: 3x1x5x6 and B: 1x4x6x7
    // 1. Reshape A to 1x15x6 (squeeze all batchdims into dim1)
    // 2. Transpose B to 6x1x4x7, Reshape to 1x6x28
    // 3. tosa.Matmul 1x15x6 1x6x28 = 1x15x28
    // 4. Reshape out to 3x5x4x7, Transpose to 3x4x5x7
    // Where there are batch dimensions that are broadcast and not, the
    // treatment is to have dim0 correspond to product of all non-broadcast
    // dimsizes:
    // Given A: 4x8x16x32 B: 8x32x17
    // 1. Reshape A to 8x64x32 (squeeze all unbroadcasted dims into dim0,
    // broadcasted dims into dim1)
    // 2. No transpose or reshape of B as its batchdims are not broadcast to.
    // 3. tosa.Matmul 8x64x32 8x32x17 = 8x64x17
    // 4. Reshape to 8x4x16x17, Transpose to 4x8x16x17

    // Check if we need to perform the broadcast on batch dim
    // Not needed if max rank < 3, or if maxrank == 3 and dim[0] matches
    auto needsBatchDimBroadcast = [&]() -> bool {
      if (maxInputRank < 3) {
        return false;
      } else {
        if (maxInputRank == 3 &&
            lhsBroadcastedShape[0] == rhsBroadcastedShape[0]) {
          return false;
        }
        return true;
      }
    };

    auto performBatchDimBroadcast = needsBatchDimBroadcast();

    // Inputs to the tosa.matmul
    Value matmulLhs, matmulRhs;

    using TensorShape_t = struct {
      int64_t dim;
      int64_t shape;
    };

    // Transpose needs to done if transposeDims are not non-monotonically
    // increasing. E.g. [0, 1, 2, 3]: No transpose [1, 0, 2, 3]: Transpose dim0
    // and dim1 The order need not be sequential, since one or more dims may
    // have been removed due to broadcasting.
    auto isTransposeRequired = [](SmallVector<int32_t> transposedDims) -> bool {
      int32_t lastDim = -1;
      for (auto &dim : transposedDims) {
        if (lastDim > dim)
          return true;
        lastDim = dim;
      }
      return false;
    };

    SmallVector<TensorShape_t> batchElems, lhsSqueezedElems, rhsSqueezedElems;

    if (!performBatchDimBroadcast) {
      // Simple with no broadcasting artifacts. Just reshape up to 3D
      matmulLhs = reshapeUpTo3DTensor(rankBroadcastedLhs);
      matmulRhs = reshapeUpTo3DTensor(rankBroadcastedRhs);

    } else {
      // In this case, either or both input matrices involve broadcasting on
      // their batch dimensions. For example:
      // 4x5x6, 1x6x7 -> 4x5x7
      // 4x1x5x6, 1x3x6x7 -> 4x3x5x7
      // Though maxInputRank is necessarily >=3 here, individual matrices may be
      // lower rank.
      // E.g. 3x4x5x6, 6 -> 3x4x5

      // These are the accumulated products of the shape of each dim:
      // 1. common dimensions: upper dimensions (dims other than two rightmost)
      // whose shapes are the same for both LHS and RHS.
      // 2. LHS squeezed dimensions: all dimensions of LHS that involve
      // broadcasting in either direction, plus the LHS[-2] shape
      // 3. RHS squeezed dimensions: all dimensions of RHS that involve
      // broadcasting in either direction, plus the RHS[-1] shape
      int64_t commonValue = 1, lhsSqueezedValue = 1, rhsSqueezedValue = 1;

      // For both LHS and RHS, the dimensions are separated into the common,
      // squeezed and remaining dim. E.g. given
      // LHS = 3x4x5x6
      // RHS = 1x4x6x7
      // common = {{dim=1, shape=4}}
      // lhs squeezed = {{dim=0, shape=3},
      //                 {dim=2, shape=5}}
      // rhs squeezed = {{dim=0, shape=1},
      //                 {dim=2, shape=7}}
      // The matmul dim is LHS[-1] and RHS[-2], i.e. 6.
      // Once this is obtained, LHS and RHS are expressed as:
      // LHS = {common, lhs_squeezed, matmul_dim}
      // RHS = {common, matmul_dim, rhs_squeezed}
      // The matmul is then performed to obtain output:
      // matmul_out = {common, lhs_squeezed, rhs_squeezed}
      // Finally, we reshape to 'unsqueeze' the LHS and RHS parts and transpose
      // them back to their correct positions.

      SmallVector<int64_t> transposedLhsShape;
      SmallVector<int32_t> transposedLhsDims;

      // Step: generate the common dim/shape information
      bool hasDynamicDims = false;
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        bool isDynamicDim = ShapedType::isDynamic(lhsBroadcastedShape[dim]);
        hasDynamicDims |= isDynamicDim;
        if (isDynamicDim ||
            lhsBroadcastedShape[dim] == rhsBroadcastedShape[dim]) {
          commonValue *= lhsBroadcastedShape[dim];
          batchElems.push_back({dim, lhsBroadcastedShape[dim]});
        }
      }
      commonValue = commonValue < 0 ? kUnknownSize : commonValue;

      // TODO: Handle the case when there are dynamic batch dimensions.
      if (hasDynamicDims)
        commonValue = kUnknownSize;

      // Step: generate the LHS squeezed dim/shape information.
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        bool isDynamicDim = ShapedType::isDynamic(lhsBroadcastedShape[dim]);
        if (!isDynamicDim &&
            lhsBroadcastedShape[dim] != rhsBroadcastedShape[dim]) {
          lhsSqueezedValue *= lhsBroadcastedShape[dim];
          lhsSqueezedElems.push_back({dim, lhsBroadcastedShape[dim]});
        }
      }
      // including LHS[-2]
      lhsSqueezedElems.push_back(
          {maxInputRank - 2, lhsBroadcastedShape[maxInputRank - 2]});
      lhsSqueezedValue *= lhsBroadcastedShape[maxInputRank - 2];
      lhsSqueezedValue = lhsSqueezedValue < 0 ? kUnknownSize : lhsSqueezedValue;

      // Step: Create the tosa.transpose array. If this array has a
      // non-monotonic series of dims, perform transpose.
      // First the common_elems
      for (uint32_t i = 0; i < batchElems.size(); i++) {
        transposedLhsShape.push_back(batchElems[i].shape);
        transposedLhsDims.push_back(batchElems[i].dim);
      }
      // then the lhs_squeezed elems
      for (uint32_t i = 0; i < lhsSqueezedElems.size(); i++) {
        transposedLhsShape.push_back(lhsSqueezedElems[i].shape);
        transposedLhsDims.push_back(lhsSqueezedElems[i].dim);
      }
      // then the final dim
      transposedLhsDims.push_back(maxInputRank - 1);
      transposedLhsShape.push_back(lhsBroadcastedShape[maxInputRank - 1]);

      bool lhsNeedsTranspose = isTransposeRequired(transposedLhsDims);

      auto lhsReshapeInput = rankBroadcastedLhs;

      if (lhsNeedsTranspose) {
        auto transposedLhsType = RankedTensorType::get(
            makeShapeLLVMCompatible(transposedLhsShape), rhsElemTy);

        std::optional<Value> transposedLhsDimsConst =
            tosa::getConstTensor<int32_t>(
                rewriter, op,
                /*vec=*/transposedLhsDims,
                /*shape=*/{static_cast<int32_t>(transposedLhsDims.size())});

        lhsReshapeInput =
            rewriter
                .create<tosa::TransposeOp>(
                    op->getLoc(),
                    OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(transposedLhsType),
                    rankBroadcastedLhs, transposedLhsDimsConst.value())
                .getResult();
      }

      // LHS = {common, lhs_squeezed, matmul_dim}
      SmallVector<int64_t> newLhsShape(
          {1, 1, lhsBroadcastedShape[maxInputRank - 1]});
      newLhsShape[0] = commonValue;
      newLhsShape[1] = hasDynamicDims ? kUnknownSize : lhsSqueezedValue;

      auto newLhsType = RankedTensorType::get(
          makeShapeLLVMCompatible(newLhsShape), lhsElemTy);

      matmulLhs = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newLhsType),
          lhsReshapeInput, rewriter.getDenseI64ArrayAttr(newLhsShape));

      SmallVector<int64_t> transposedRhsShape;
      SmallVector<int32_t> transposedRhsDims;

      // Step: Create the RHS transpose sequence
      // RHS = {common, matmul_dim, rhs_squeezed}
      // first the common_dims
      for (uint32_t i = 0; i < batchElems.size(); i++) {
        transposedRhsShape.push_back(batchElems[i].shape);
        transposedRhsDims.push_back(batchElems[i].dim);
      }
      // The matmul_dim of RHS
      transposedRhsDims.push_back(maxInputRank - 2);
      transposedRhsShape.push_back(rhsBroadcastedShape[maxInputRank - 2]);
      // finally all the rhs_squeeze dims
      hasDynamicDims = false;
      for (uint32_t dim = 0; dim < maxInputRank - 2; dim++) {
        bool isDynamicDim = ShapedType::isDynamic(rhsBroadcastedShape[dim]);
        hasDynamicDims |= isDynamicDim;
        if (!isDynamicDim &&
            rhsBroadcastedShape[dim] != lhsBroadcastedShape[dim]) {
          rhsSqueezedElems.push_back({dim, rhsBroadcastedShape[dim]});
          rhsSqueezedValue *= rhsBroadcastedShape[dim];
        }
      }
      rhsSqueezedElems.push_back(
          {maxInputRank - 1, rhsBroadcastedShape[maxInputRank - 1]});
      rhsSqueezedValue *= rhsBroadcastedShape[maxInputRank - 1];
      for (uint32_t i = 0; i < rhsSqueezedElems.size(); i++) {
        transposedRhsShape.push_back(rhsSqueezedElems[i].shape);
        transposedRhsDims.push_back(rhsSqueezedElems[i].dim);
      }

      auto transposedRhsType = RankedTensorType::get(
          makeShapeLLVMCompatible(transposedRhsShape), rhsElemTy);

      if (hasDynamicDims)
        rhsSqueezedValue = kUnknownSize;

      SmallVector<int64_t> newRhsShape(
          {commonValue < 0 ? kUnknownSize : commonValue,
           rhsBroadcastedShape[maxInputRank - 2], rhsSqueezedValue});
      auto newRhsType = RankedTensorType::get(
          makeShapeLLVMCompatible(newRhsShape), rhsElemTy);

      bool rhsNeedsTranspose = isTransposeRequired(transposedRhsDims);

      auto transposedRhsValue = rankBroadcastedRhs;

      if (rhsNeedsTranspose) {
        std::optional<Value> transposedRhsDimsConst =
            tosa::getConstTensor<int32_t>(
                rewriter, op,
                /*vec=*/transposedRhsDims,
                /*shape=*/{static_cast<int32_t>(transposedRhsDims.size())});

        transposedRhsValue =
            rewriter
                .create<tosa::TransposeOp>(
                    op->getLoc(),
                    OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(transposedRhsType),
                    rankBroadcastedRhs, transposedRhsDimsConst.value())
                .getResult();
      }

      // reshape
      matmulRhs = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              newRhsType),
          transposedRhsValue, rewriter.getDenseI64ArrayAttr(newRhsShape));
    }

    auto matmulLhsShape = makeShapeTorchCompatible(
        cast<RankedTensorType>(matmulLhs.getType()).getShape());
    auto matmulRhsShape = makeShapeTorchCompatible(
        cast<RankedTensorType>(matmulRhs.getType()).getShape());

    // The reshape/transpose should ensure the tosa.matmul always has same
    // batch size for either matrix. If if shapes are dynamic, they'll be
    // appropriately handled.
    assert(matmulLhsShape[0] == matmulRhsShape[0] &&
           "tosa.matmul needs same batchsize on LHS and RHS");

    SmallVector<int64_t> matmulOutputShape(
        {matmulLhsShape[0], matmulLhsShape[1], matmulRhsShape[2]});

    auto mmOutputTy = RankedTensorType::get(
        makeShapeLLVMCompatible(matmulOutputShape), outputElemTy);
    auto mmOpResult =
        rewriter
            .create<tosa::MatMulOp>(
                op->getLoc(),
                OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
                    mmOutputTy),
                matmulLhs, matmulRhs)
            .getResult();

    auto torchOpOutputType = lhsTy.getElementType();
    auto castOutputTy = RankedTensorType::get(
        makeShapeLLVMCompatible(matmulOutputShape), torchOpOutputType);
    auto castResult = rewriter.createOrFold<tosa::CastOp>(
        op->getLoc(),
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            castOutputTy),
        mmOpResult);

    // Perform the reshape to output shape. This is always required unless max
    // input rank=3 and there was no broadcasting, in which case the tosa.matmul
    // output itself is correctly shaped.
    bool performOpReshape = !(maxInputRank == 3 && !performBatchDimBroadcast);

    if (performOpReshape) {
      // Since the output shape may be unknown, we construct it
      // independently and reshape. Otherwise reshape may be expressed for
      // an unknown to-be-inferred output shape. The final tensor.cast
      // reshapes the known shape to the desired output shape.
      auto computeOpShape = [&](SmallVector<int64_t> &reshapedOpShape,
                                SmallVector<int32_t> &transposedOpDims,
                                SmallVector<int64_t> &transposedOpShapes) {
        if (maxInputRank == 1)
          return;

        if (maxInputRank == 2) {
          if (lhsRank == 2)
            reshapedOpShape.push_back(lhsShape[0]);
          if (rhsRank == 2)
            reshapedOpShape.push_back(rhsShape[1]);
          return;
        }

        // Step: Construct the output transpose/reshape information
        // First the common_dims
        for (uint32_t i = 0; i < batchElems.size(); i++) {
          reshapedOpShape.push_back(batchElems[i].shape);
          transposedOpDims.push_back(batchElems[i].dim);
        }

        // Then the LHS squeezed dims
        for (uint32_t i = 0; i < lhsSqueezedElems.size() - 1; i++) {
          // Only dims that don't broadcast - broadcasting ones come from the
          // other input.
          if (lhsSqueezedElems[i].shape != 1) {
            reshapedOpShape.push_back(lhsSqueezedElems[i].shape);
            transposedOpDims.push_back(lhsSqueezedElems[i].dim);
          }
        }
        // The last squeezed dim is lhs[-2] which needs to be
        // checked separately for broadcasting
        if (lhsRank > 1) {
          reshapedOpShape.push_back(lhsBroadcastedShape[maxInputRank - 2]);
          transposedOpDims.push_back(maxInputRank - 2);
        }

        // then the RHS squeezed dims except rhs[-1] which is handled like
        // lhs[-2]
        for (uint32_t i = 0; i < rhsSqueezedElems.size() - 1; i++) {
          if (rhsSqueezedElems[i].shape != 1) {
            reshapedOpShape.push_back(rhsSqueezedElems[i].shape);
            transposedOpDims.push_back(rhsSqueezedElems[i].dim);
          }
        }
        // rhs[-1]
        if (rhsRank > 1) {
          reshapedOpShape.push_back(rhsBroadcastedShape[maxInputRank - 1]);
          transposedOpDims.push_back(maxInputRank - 1);
        }

        // The transposition order is the inverse of what we actually want,
        // inversing should fix this:
        llvm::SmallVector<int> inverseTransposeDims(transposedOpDims.size());
        for (int i = 0, s = transposedOpDims.size(); i < s; ++i)
          inverseTransposeDims[transposedOpDims[i]] = i;

        transposedOpDims = inverseTransposeDims;

        // Final transposed output shape construction
        for (uint32_t i = 0; i < maxInputRank - 2; i++) {
          if (lhsBroadcastedTy.isDynamicDim(i)) {
            transposedOpShapes.push_back(kUnknownSize);
          } else {
            if (lhsBroadcastedShape[i] == rhsBroadcastedShape[i]) {
              transposedOpShapes.push_back(lhsBroadcastedShape[i]);
            } else {
              transposedOpShapes.push_back(lhsBroadcastedShape[i] == 1
                                               ? rhsBroadcastedShape[i]
                                               : lhsBroadcastedShape[i]);
            }
          }
        }
        if (lhsRank > 1)
          transposedOpShapes.push_back(lhsBroadcastedShape[maxInputRank - 2]);
        if (rhsRank > 1)
          transposedOpShapes.push_back(rhsBroadcastedShape[maxInputRank - 1]);

        return;
      };

      SmallVector<int64_t> reshapedOpShape, transposedOpShape;
      SmallVector<int32_t> transposedOpDims;

      computeOpShape(reshapedOpShape, transposedOpDims, transposedOpShape);

      bool opNeedsTranspose = isTransposeRequired(transposedOpDims);

      // Perform reshape
      auto reshapedOpType = RankedTensorType::get(
          makeShapeLLVMCompatible(reshapedOpShape), torchOpOutputType);
      auto reshapedOp = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              reshapedOpType),
          castResult, rewriter.getDenseI64ArrayAttr(reshapedOpShape));

      if (opNeedsTranspose) {

        std::optional<Value> transposedOpShapeConst =
            tosa::getConstTensor<int32_t>(
                rewriter, op,
                /*vec=*/transposedOpDims,
                /*shape=*/{static_cast<int32_t>(transposedOpDims.size())});

        auto transposedOpType = RankedTensorType::get(
            makeShapeLLVMCompatible(transposedOpShape), torchOpOutputType);
        output = rewriter
                     .create<tosa::TransposeOp>(
                         op->getLoc(),
                         OpConversionPattern<AtenOpT>::getTypeConverter()
                             ->convertType(transposedOpType),
                         reshapedOp.getResult(), transposedOpShapeConst.value())
                     .getResult();

      } else {
        output = reshapedOp.getResult();
      }
    } else {
      output = castResult;
    }

    return success();
  }
  // The default version just reads two inputs, computes output and returns it.
  // Other versions may add a bias, apply GEMM-style alpha/beta scaling etc.
  virtual LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value lhs, rhs;

    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "Failed to read matmul inputs");

    Value output;

    if (failed(performMatmul(op, adaptor, rewriter, lhs, rhs, output)))
      return rewriter.notifyMatchFailure(op,
                                         "Failed to perform matmul operation");

    rewriter.replaceOp(op, output);
    return success();
  }
};

// Legalizes the torch.matmul op for general n-dim matmul.
template <typename AtenOpT>
class ConvertAtenMatMulOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {
    lhs = adaptor.getSelf();
    auto lhsTy = cast<RankedTensorType>(lhs.getType());

    rhs = adaptor.getOther();
    auto rhsTy = cast<RankedTensorType>(rhs.getType());

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA matmul");

    return success();
  }
};

// Implements handling of aten.mm and aten.bmm ops.
template <typename AtenOpT>
class ConvertAtenMmOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {

    lhs = adaptor.getSelf();
    auto lhsTy = cast<RankedTensorType>(lhs.getType());

    rhs = adaptor.getMat2();
    auto rhsTy = cast<RankedTensorType>(rhs.getType());

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (isa<AtenMmOp>(op)) {
      // Mm takes two 2D tensors.
      if (lhsRank != 2 || rhsRank != 2)
        return op.emitError("aten.mm called but matrix rank != 2");
    } else if (isa<AtenBmmOp>(op)) {
      // Bmm takes two 3D tensors.
      if (lhsRank != 3 || rhsRank != 3)
        return op.emitError("aten.bmm called but matrix rank != 3");
    }

    return success();
  }
};

// Implements handling of aten.linear op.
template <typename AtenOpT>
class ConvertAtenLinearOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(AtenOpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {

    lhs = adaptor.getInput();
    auto lhsTy = cast<RankedTensorType>(lhs.getType());

    rhs = adaptor.getWeight();
    auto rhsTy = cast<RankedTensorType>(rhs.getType());

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (lhsRank != 2 && lhsRank != 3)
      return op.emitError("aten.Linear called but input rank not 2 or 3");
    if (rhsRank != 2 && rhsRank != 3)
      return op.emitError("aten.Linear called but weight rank not 2 or 3");

    // Protection against crash due to unguarded code in TOSA->LinAlg.
    // TODO: This should be handled in TOSA->LinAlg instead.
    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "aten.Linear needs statically shaped input");

    return success();
  }
  // Override the default rewriter to perform RHS transpose and bias addition as
  // well.
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value lhs, rhs;

    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "Failed to read matmul inputs");

    // The aten.Linear op has a bias tensor that is added to the matmul output.
    auto bias = adaptor.getBias();
    auto biasTy = bias.getType();

    // TOSA does not mandate that elementwise op tensors need to be ranked.
    if (!isa<Torch::NoneType>(biasTy) && !isa<TensorType>(biasTy))
      return rewriter.notifyMatchFailure(
          op, "Only tensor types supported in GEMM to TOSA for bias tensor");

    // RHS must have its last two dims transposed prior to matrix
    // multiplication.
    auto rhsTy = cast<RankedTensorType>(rhs.getType());
    auto rhsRank = rhsTy.getRank();
    auto rhsShape = makeShapeTorchCompatible(rhsTy.getShape());
    auto rhsElemTy = rhsTy.getElementType();

    // Create a non-const shape array to transpose dims.
    SmallVector<int64_t> transposedRhsShape;
    for (auto &shape : rhsShape)
      transposedRhsShape.push_back(shape);
    SmallVector<int32_t> transposedRhsDims;
    for (int32_t i = 0; i < rhsRank; i++)
      transposedRhsDims.push_back(i);

    // Swap the last two dims.
    std::swap(transposedRhsShape[rhsRank - 1], transposedRhsShape[rhsRank - 2]);
    std::swap(transposedRhsDims[rhsRank - 1], transposedRhsDims[rhsRank - 2]);

    std::optional<Value> transposedRhsShapeConst =
        tosa::getConstTensor<int32_t>(
            rewriter, op,
            /*vec=*/transposedRhsDims,
            /*shape=*/{static_cast<int32_t>(transposedRhsDims.size())});

    auto transposedRhsType = RankedTensorType::get(
        makeShapeLLVMCompatible(transposedRhsShape), rhsElemTy);
    rhs = rewriter.create<tosa::TransposeOp>(
        op->getLoc(),
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            transposedRhsType),
        rhs, transposedRhsShapeConst.value());

    Value matmulOutput;
    if (failed(
            this->performMatmul(op, adaptor, rewriter, lhs, rhs, matmulOutput)))
      return rewriter.notifyMatchFailure(op,
                                         "Failed to perform matmul operation");

    Value matmulPlusBias = matmulOutput;
    if (!isa<Torch::NoneType>(biasTy)) {
      // Bias addition broadcasts to the matmul output shape.
      matmulPlusBias =
          rewriter
              .create<tosa::AddOp>(op->getLoc(), matmulOutput.getType(),
                                   matmulOutput, bias)
              .getResult();
    }
    rewriter.replaceOp(op, matmulPlusBias);
    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenRsubScalarOp>::matchAndRewrite(
    AtenRsubScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto self = adaptor.getSelf();
  auto otherScalar = op.getOther();
  auto alphaScalar = op.getAlpha();

  auto selfTy = cast<RankedTensorType>(self.getType());
  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in TOSA Rsub");

  if (!isa<mlir::FloatType>(selfTy.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization supported");

  Value otherTensor, alphaTensor;

  if (failed(torchScalarToTosaTensor(rewriter, op, otherScalar, otherTensor,
                                     selfTy.getElementType(), {})))
    return rewriter.notifyMatchFailure(
        op, "Currently only scalar constants are supported for "
            "conversion in TOSA Rsub operation");

  if (failed(torchAlphaToTosaTensor(rewriter, op.getOperation(), alphaScalar,
                                    alphaTensor, selfTy.getElementType(),
                                    /*checkForUnity=*/true)))
    return failure();

  auto multTensor = rewriter.create<tosa::MulOp>(
      op->getLoc(), getTypeConverter()->convertType(op.getType()), self,
      alphaTensor, /*shift=*/0);

  rewriter.replaceOpWithNewOp<tosa::SubOp>(
      op, getTypeConverter()->convertType(op.getType()), otherTensor,
      multTensor);

  return success();
}

/// tosa.conv2d does not support group convolution.
/// Therefore, we create multiple ops where the input, kernel
/// and bias are slices of the original inputs.
/// Afterwards we concat the results into a single tensor.
/// This is inspired by the legalization done in onnx-mlir.
Value createConvInGroups(PatternRewriter &rewriter, Operation *op,
                         Type &resultType,
                         const llvm::ArrayRef<int64_t> weightShape,
                         Value &input, Value &weights, Value &bias,
                         const int64_t groups, DenseI64ArrayAttr &pads,
                         DenseI64ArrayAttr &strides,
                         DenseI64ArrayAttr &dilations) {
  // Set up constants outside of loop
  const int64_t sizeOfSliceInput = weightShape[1];
  const int64_t sizeOfSliceKernel = weightShape[0] / groups;
  auto inputShape = cast<ShapedType>(input.getType()).getShape();

  llvm::SmallVector<int64_t, 4> inputSize = {inputShape[0], inputShape[1],
                                             inputShape[2], sizeOfSliceInput};
  llvm::SmallVector<int64_t, 4> kernelSize = {sizeOfSliceKernel, weightShape[2],
                                              weightShape[3], weightShape[1]};
  llvm::SmallVector<Value> sliceValues;
  Type outputType = RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(4, ShapedType::kDynamic),
      cast<ShapedType>(resultType).getElementType());
  for (int64_t i = 0; i < groups; i++) {
    // Slice input
    Value sliceInput = tosa::buildSlice(
        rewriter, input, {0, 0, 0, i * sizeOfSliceInput}, inputSize);

    // Slice kernel
    Value sliceWeight = tosa::buildSlice(
        rewriter, weights, {i * sizeOfSliceKernel, 0, 0, 0}, kernelSize);

    // Slice bias
    Value sliceBias = tosa::buildSlice(rewriter, bias, {i * sizeOfSliceKernel},
                                       {sizeOfSliceKernel});

    // Create conv
    Value tempConv2D = tosa::CreateOpAndInfer<mlir::tosa::Conv2DOp>(
        rewriter, input.getLoc(), outputType, sliceInput, sliceWeight,
        sliceBias, pads, strides, dilations);
    // Add value to vector
    sliceValues.push_back(tempConv2D);
  }

  constexpr int64_t channelDim = 3;
  // Create concat op
  return tosa::CreateOpAndInfer<mlir::tosa::ConcatOp>(
      rewriter, op->getLoc(), outputType, sliceValues, channelDim);
}

template <>
LogicalResult ConvertAtenOp<AtenConvolutionOp>::matchAndRewrite(
    AtenConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  bool transposed;
  if (!matchPattern(op.getTransposed(), m_TorchConstantBool(&transposed)))
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: non-constant value for transposed not supported");
  if (transposed)
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: transposed convolution not supported");

  auto input = adaptor.getInput();
  auto weight = adaptor.getWeight();

  auto inputTy = cast<RankedTensorType>(input.getType());
  auto weightTy = cast<RankedTensorType>(weight.getType());
  auto outputTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

  if (!inputTy || !weightTy || !outputTy)
    return rewriter.notifyMatchFailure(
        op, "Input, weight and output to Convolution must be ranked tensors");

  auto inputElemTy = inputTy.getElementType();
  auto weightElemTy = weightTy.getElementType();
  auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
  auto weightShape = makeShapeTorchCompatible(weightTy.getShape());

  if (inputTy.getRank() != 4)
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: only 2D convolutions supported");

  if (!weightTy.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: TOSA only supports static weight");

  // Bias is optional. TOSA mandates a zero tensor here, so construct one if
  // required.
  auto bias = adaptor.getBias();
  if (isa<Torch::NoneType>(adaptor.getBias().getType())) {
    // TBD: This is only valid for quantized 8-bit. For 16-bit, the bias (and
    // accumulator) are 48-bit and not 32-bit, and requires the use of APInt to
    // define a 48-bit int.
    if (isa<quant::QuantizedType>(inputElemTy)) {
      SmallVector<int32_t> zeroVec(weightShape[0], 0);
      bias = tosa::getConstTensor<int32_t>(
                 rewriter, op, zeroVec, {static_cast<int32_t>(weightShape[0])})
                 .value();
    } else {
      SmallVector<float> zeroVec(weightShape[0], 0);
      bias = tosa::getConstTensor<float>(rewriter, op, zeroVec,
                                         {static_cast<int32_t>(weightShape[0])},
                                         inputElemTy)
                 .value();
    }
  } else {
    if (!cast<RankedTensorType>(bias.getType()))
      return rewriter.notifyMatchFailure(
          op, "Bias provided but not a ranked tensor");
  }
  auto biasElemTy =
      isa<mlir::FloatType>(inputElemTy) ? inputElemTy : rewriter.getI32Type();

  int64_t groups;
  if (!matchPattern(op.getGroups(), m_TorchConstantInt(&groups))) {
    return rewriter.notifyMatchFailure(op, "non-const group size unsupported");
  }

  SmallVector<int64_t, 2> stride;
  if (!matchPattern(adaptor.getStride(), m_TorchListOfConstantInts(stride)))
    return rewriter.notifyMatchFailure(op, "non-const stride list unsupported");

  SmallVector<int64_t, 2> padding_2d;
  if (!matchPattern(adaptor.getPadding(),
                    m_TorchListOfConstantInts(padding_2d)))
    return rewriter.notifyMatchFailure(op,
                                       "non-const padding list unsupported");

  // TOSA uses 4D padding {t, b, l, r} while Torch defines 2D padding {t, l}.
  // The Torch OFM computation uses 2*pad in each spatial direction, implying
  // the same t=b and l=r values for TOSA.
  SmallVector<int64_t> padding(
      {padding_2d[0], padding_2d[0], padding_2d[1], padding_2d[1]});

  SmallVector<int64_t, 2> dilation;
  if (!matchPattern(adaptor.getDilation(), m_TorchListOfConstantInts(dilation)))
    return rewriter.notifyMatchFailure(op,
                                       "non-const dilation list unsupported");

  // TOSA works in NHWC and takes OHWI (conv) / HWIM (depthwise conv) weights.
  // Perform the necessary transformations.
  std::optional<Value> nchwToNhwcTransposeConst =
      tosa::getConstTensor<int32_t>(rewriter, op,
                                    /*vec=*/{0, 2, 3, 1},
                                    /*shape=*/{static_cast<int32_t>(4)});
  SmallVector<int64_t> transposedInputShape(
      {inputShape[0], inputShape[2], inputShape[3], inputShape[1]});
  auto transposedInputType = RankedTensorType::get(
      makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
  auto transposedInput =
      rewriter
          .create<tosa::TransposeOp>(
              op->getLoc(),
              getTypeConverter()->convertType(transposedInputType), input,
              nchwToNhwcTransposeConst.value())
          .getResult();

  SmallVector<int64_t> transformedWeightShape;
  RankedTensorType transformedWeightType;
  Value transformedWeight;
  int64_t outputCDim;
  if (groups == 1 || weightShape[1] != 1) {
    // full (group) convolution: O(I/G)HW-> OHWI
    transformedWeightShape = {weightShape[0], weightShape[2], weightShape[3],
                              weightShape[1]};
    transformedWeightType = RankedTensorType::get(
        makeShapeLLVMCompatible(transformedWeightShape), weightElemTy);
    transformedWeight =
        rewriter
            .create<tosa::TransposeOp>(
                op->getLoc(),
                getTypeConverter()->convertType(transformedWeightType), weight,
                nchwToNhwcTransposeConst.value())
            .getResult();
    outputCDim = transformedWeightShape[0];
  } else {
    // depthwise convolution: O(I/G)HW-> HWIM)
    // transpose: O(I/G)HW -> HWO(I/G)
    std::optional<Value> transposeConst =
        tosa::getConstTensor<int32_t>(rewriter, op,
                                      /*vec=*/{2, 3, 0, 1},
                                      /*shape=*/{static_cast<int32_t>(4)});
    SmallVector<int64_t> transposedWeightShape = {
        weightShape[2], weightShape[3], weightShape[0], weightShape[1]};
    auto transposedWeightType = RankedTensorType::get(
        makeShapeLLVMCompatible(transposedWeightShape), weightElemTy);
    auto transposedWeight =
        rewriter
            .create<tosa::TransposeOp>(
                op->getLoc(),
                getTypeConverter()->convertType(transposedWeightType), weight,
                transposeConst.value())
            .getResult();

    // reshape: HWO(I/G) -> HWIM
    outputCDim = makeShapeTorchCompatible(outputTy.getShape())[1];
    if (outputCDim == kUnknownSize) {
      return rewriter.notifyMatchFailure(
          op, "number of output channels must be statically known for "
              "depthwise convolutions");
    }
    transformedWeightShape = {
        transposedWeightShape[0],
        transposedWeightShape[1],
        groups,
        outputCDim / groups,
    };
    transformedWeightType = RankedTensorType::get(
        makeShapeLLVMCompatible(transformedWeightShape), weightElemTy);
    transformedWeight =
        rewriter
            .create<tosa::ReshapeOp>(
                op->getLoc(),
                getTypeConverter()->convertType(transformedWeightType),
                transposedWeight,
                rewriter.getDenseI64ArrayAttr(transformedWeightShape))
            .getResult();
  }

  int64_t outputHDim, outputWDim;
  if (inputTy.hasStaticShape()) {
    int64_t inputHDim = inputShape[2];
    int64_t inputWDim = inputShape[3];
    int64_t weightHDim = weightShape[2];
    int64_t weightWDim = weightShape[3];
    outputHDim = (inputHDim + padding[0] + padding[1] -
                  dilation[0] * (weightHDim - 1) - 1) /
                     stride[0] +
                 1;
    outputWDim = (inputWDim + padding[2] + padding[3] -
                  dilation[1] * (weightWDim - 1) - 1) /
                     stride[1] +
                 1;
  } else {
    outputHDim = kUnknownSize;
    outputWDim = kUnknownSize;
  }

  // Output shape is NHWC, to be transposed back to NCHW. Output elemTy for
  // quantized input is i32, which gets rescaled down to quantized output range.
  SmallVector<int64_t> outputShape = {transposedInputShape[0], outputHDim,
                                      outputWDim, outputCDim};

  DenseI64ArrayAttr paddingAttr = rewriter.getDenseI64ArrayAttr(padding);
  DenseI64ArrayAttr strideAttr = rewriter.getDenseI64ArrayAttr(stride);
  DenseI64ArrayAttr dilationAttr = rewriter.getDenseI64ArrayAttr(dilation);

  Value convOpResult;
  if (groups == 1) {
    // full convolution
    auto convOpTy =
        RankedTensorType::get(makeShapeLLVMCompatible(outputShape), biasElemTy);
    convOpResult =
        rewriter
            .create<tosa::Conv2DOp>(op->getLoc(),
                                    getTypeConverter()->convertType(convOpTy),
                                    transposedInput, transformedWeight, bias,
                                    paddingAttr, strideAttr, dilationAttr)
            .getResult();
  } else if (weightShape[1] == 1) {
    // depthwise convolution
    auto convOpTy =
        RankedTensorType::get(makeShapeLLVMCompatible(outputShape), biasElemTy);
    convOpResult =
        rewriter
            .create<tosa::DepthwiseConv2DOp>(
                op->getLoc(), getTypeConverter()->convertType(convOpTy),
                transposedInput, transformedWeight, bias, paddingAttr,
                strideAttr, dilationAttr)
            .getResult();
  } else {
    // general group convolution
    convOpResult = createConvInGroups(
        rewriter, op, outputTy, weightShape, transposedInput, transformedWeight,
        bias, groups, paddingAttr, strideAttr, dilationAttr);
  }

  std::optional<Value> nhwcToNchwTransposeConst =
      tosa::getConstTensor<int32_t>(rewriter, op,
                                    /*vec=*/{0, 3, 1, 2},
                                    /*shape=*/{static_cast<int32_t>(4)});
  SmallVector<int64_t> transposedOutputShape(
      {outputShape[0], outputShape[3], outputShape[1], outputShape[2]});
  auto transposedOutputType = RankedTensorType::get(
      makeShapeLLVMCompatible(transposedOutputShape), biasElemTy);
  auto transposedOutput =
      rewriter
          .create<tosa::TransposeOp>(
              op->getLoc(),
              getTypeConverter()->convertType(transposedOutputType),
              convOpResult, nhwcToNchwTransposeConst.value())
          .getResult();

  Value rescaledResult = transposedOutput;
  if (isa<quant::QuantizedType>(inputElemTy)) {
    rescaledResult = tosa::buildRescaleOpConvOutput(
        rewriter, op, transposedOutput, inputTy, weightTy, outputTy);
  }

  rewriter.replaceOpWithNewOp<tensor::CastOp>(
      op, getTypeConverter()->convertType(op.getType()), rescaledResult);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenReshapeOp>::matchAndRewrite(
    AtenReshapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto self = adaptor.getSelf();

  auto selfTy = cast<RankedTensorType>(self.getType());
  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in TOSA Reshape");

  // Check that at most one dimension is -1
  SmallVector<int64_t> newShape;
  if (!matchPattern(op.getShape(), m_TorchListOfConstantInts(newShape)))
    return rewriter.notifyMatchFailure(
        op, "Only constant shape supported in TOSA Reshape");

  int auto_sz = 0;
  for (auto s : newShape)
    auto_sz += (s == -1 ? 1 : 0);
  if (auto_sz > 1)
    return rewriter.notifyMatchFailure(
        op, "At most one dimension may be specified as -1 to "
            "automatically calculate its size");

  auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                       selfTy.getElementType());

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(newType), self,
      rewriter.getDenseI64ArrayAttr(newShape));

  return success();
}

Value computeBatchNorm(Operation *op, ConversionPatternRewriter &rewriter,
                       Type outType, Value input, Value variance, Value eps,
                       Value mean, Value weight, Value bias) {
  // For PyTorch:
  //   scale  = gamma = weight
  //   offset = beta  = bias
  // Lowering:
  // fused batchnorm = (input-mean) * scale * rsqrt(var+epsilon)) + offset
  //
  // shape_0 = ones(input.rank)
  // shape_0[input.rank-1] = input.shape[input.rank-1]
  // shape_1 = ones(1)
  //
  // bmean  = reshape(mean, shape_0)
  // bscale = reshape(scale, shape_0)
  // boffset= reshape(offset, shape_0)
  // beps   = reshape(epsilon, shape_1)
  //
  // op1 = sub(input, bmean)
  // op2 = add(var, beps)
  // op3 = rsqrt(op2)
  // bvar = reshape(op3, shape_0)
  // op4 = mul(op1, bvar)
  // op5 = mul(op4, bscale)
  // op6 = add(op5, boffset)

  auto op1SubInputMean =
      rewriter.create<tosa::SubOp>(op->getLoc(), outType, input, mean);

  auto op2AddVarEpsilon = rewriter.create<tosa::AddOp>(
      op->getLoc(), variance.getType(), variance, eps);

  auto op3RsqrtOp2 = rewriter.create<tosa::RsqrtOp>(
      op->getLoc(), variance.getType(), op2AddVarEpsilon.getResult());

  auto op4MulOp1Op3 = rewriter.create<tosa::MulOp>(op->getLoc(), outType,
                                                   op1SubInputMean.getResult(),
                                                   op3RsqrtOp2.getResult(), 0);

  auto op5MulOp4Scale = rewriter.create<tosa::MulOp>(
      op->getLoc(), outType, op4MulOp1Op3.getResult(), weight, 0);

  return rewriter
      .create<tosa::AddOp>(op->getLoc(), outType, op5MulOp4Scale.getResult(),
                           bias)
      .getResult();
}

// This lowering is based on the TensorFlow to TOSA lowering.
template <>
LogicalResult ConvertAtenOp<AtenBatchNormOp>::matchAndRewrite(
    AtenBatchNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor output
  if (!dyn_cast<RankedTensorType>(adaptor.getInput().getType()))
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types are supported");

  auto outType = getTypeConverter()->convertType(op.getType());

  // Note: cudnn_enabled is not handled.

  // FIXME: Handle training and momentum.
  if (isa<Torch::NoneType>(op.getMomentum().getType()))
    return rewriter.notifyMatchFailure(op, "Unsupported None for momentum");

  auto meanType = dyn_cast<TensorType>(adaptor.getRunningMean().getType());
  auto varianceType = dyn_cast<TensorType>(adaptor.getRunningVar().getType());
  if (!varianceType || !meanType)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types are supported");

  // Normalization ops perform elementwise ops of a single mean/stdev value
  // against the feature map and because input is NCHW, the rank-1 value must be
  // reshaped so it sits on the same dim as 'C'.
  auto reshapeToNormInputDim = [&](Operation *op,
                                   ConversionPatternRewriter &rewriter,
                                   const TypeConverter *converter, Type outType,
                                   const Value toBcast, Value &result) {
    RankedTensorType toBcastType =
        dyn_cast<RankedTensorType>(toBcast.getType());
    if (toBcastType.getRank() > 1)
      return rewriter.notifyMatchFailure(op, "Rank cannot be more than 1");

    RankedTensorType outTensorType = cast<RankedTensorType>(outType);
    SmallVector<int64_t> newShape = {
        makeShapeTorchCompatible(toBcastType.getShape())[0]};
    for (auto i = 2; i < outTensorType.getRank(); ++i)
      newShape.push_back(1);
    auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                         outTensorType.getElementType());

    result = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(), newType, toBcast,
        rewriter.getDenseI64ArrayAttr(newShape));

    return success();
  };

  Value meanVal, varianceVal, weightVal, biasVal;
  assert(meanType.getNumElements() != 0 && varianceType.getNumElements() != 0);
  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.getRunningMean(), meanVal)))
    return rewriter.notifyMatchFailure(op, "Failed to reshape running mean");

  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.getRunningVar(), varianceVal)))
    return rewriter.notifyMatchFailure(op,
                                       "Failed to reshape running variance");

  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.getWeight(), weightVal)))
    return rewriter.notifyMatchFailure(op, "Failed to reshape weight");

  if (failed(reshapeToNormInputDim(op.getOperation(), rewriter,
                                   getTypeConverter(), outType,
                                   adaptor.getBias(), biasVal)))
    return rewriter.notifyMatchFailure(op, "Failed to reshape bias");

  double eps;
  if (!matchPattern(op.getEps(), m_TorchConstantFloat(&eps)))
    return rewriter.notifyMatchFailure(op, "eps must be a scalar constant");

  auto epsilonConst = tosa::getConstTensor<float>(rewriter, op.getOperation(),
                                                  {static_cast<float>(eps)}, {},
                                                  meanType.getElementType())
                          .value();

  auto batchNorm =
      computeBatchNorm(op, rewriter, outType, adaptor.getInput(), varianceVal,
                       epsilonConst, meanVal, weightVal, biasVal);

  rewriter.replaceOp(op, {batchNorm});

  return success();
}

// This lowering is loosely based on Torch to LinAlg lowering.
template <>
LogicalResult ConvertAtenOp<AtenNativeLayerNormOp>::matchAndRewrite(
    AtenNativeLayerNormOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // The key difference from BatchNorm is that a specified set of dims
  // (normalized_shape) are chosen to compute the mean and variance from input.
  // Where as in BatchNorm the mean and variance are operands. tosa::ReduceSumOp
  // is used to sum up the these dims for mean and for variance. The results
  // eventually being reshaped for broadcasting.

  // Not a ranked tensor output
  if (!dyn_cast<RankedTensorType>(adaptor.getInput().getType()))
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types are supported");

  auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
  if (inputType.getRank() > 4)
    return rewriter.notifyMatchFailure(op,
                                       "Only up to 4D tensors are supported");

  auto outType = getTypeConverter()->convertType(op.getType(0));

  // Note: cudnn_enabled is not handled.

  // FIXME: Handle the None cases for the optional parameters.
  if (isa<Torch::NoneType>(adaptor.getWeight().getType()))
    return rewriter.notifyMatchFailure(op, "Unsupported None for weight");
  if (isa<Torch::NoneType>(adaptor.getBias().getType()))
    return rewriter.notifyMatchFailure(op, "Unsupported None for bias");

  auto weightType = cast<RankedTensorType>(adaptor.getWeight().getType());
  auto biasType = cast<RankedTensorType>(adaptor.getBias().getType());
  int64_t inputRank = inputType.getRank();
  Type elemTy = inputType.getElementType();
  SmallVector<int64_t> inputTypeShape(
      makeShapeTorchCompatible(inputType.getShape()));

  // Check if all the arguments meet the requirements.
  SmallVector<int64_t> normalizedShapeSizesInt;
  if (!matchPattern(op.getNormalizedShape(),
                    m_TorchListOfConstantInts(normalizedShapeSizesInt))) {
    return rewriter.notifyMatchFailure(op, "Unimplemented normalized_shape not"
                                           "constructed from ListConstruct");
  }
  int64_t normalizedShapeRank = normalizedShapeSizesInt.size();
  if (weightType.getRank() != normalizedShapeRank ||
      biasType.getRank() != normalizedShapeRank ||
      inputRank < normalizedShapeRank || normalizedShapeRank < 1)
    return rewriter.notifyMatchFailure(op, "Input or weight or bias shape or"
                                           "normalized shape not compatible");

  // Check all the dimensions match the normalized_shape, only static shapes as
  // of now
  int64_t meanAndVarShapeRank = inputRank - normalizedShapeSizesInt.size();
  for (auto en : llvm::enumerate((normalizedShapeSizesInt))) {
    int64_t index = en.index();
    int64_t value = en.value();
    if (inputTypeShape[index + meanAndVarShapeRank] != value ||
        makeShapeTorchCompatible(weightType.getShape())[index] != value ||
        makeShapeTorchCompatible(biasType.getShape())[index] != value)
      return rewriter.notifyMatchFailure(op,
                                         "mismatching contracting dimension");
  }

  // Helper for computing mean and variance.
  auto computeSumAndReshape = [&](Value toReduce, RankedTensorType toReduceType,
                                  Type outType, SmallVector<int64_t> outShape) {
    Value sumDiv = toReduce;
    SmallVector<int64_t> toReduceShape(
        makeShapeTorchCompatible(toReduceType.getShape()));
    for (int64_t i = toReduceShape.size() - 1; i >= meanAndVarShapeRank; i--) {
      toReduceShape[i] = 1;
      sumDiv = rewriter.create<tosa::ReduceSumOp>(
          op.getLoc(),
          RankedTensorType::get(makeShapeLLVMCompatible(toReduceShape),
                                inputType.getElementType()),
          sumDiv, rewriter.getI32IntegerAttr(i));
    }

    return rewriter.create<tosa::ReshapeOp>(
        op.getLoc(), outType, sumDiv, rewriter.getDenseI64ArrayAttr(outShape));
  };

  // TOSA has integer Div so, compute reciprocal of element count to be used in
  // mul.
  int64_t elemCnt = 1;
  for (auto i : normalizedShapeSizesInt)
    elemCnt *= i;

  auto elemCntConst =
      tosa::getConstTensor<float>(rewriter, op.getOperation(),
                                  {static_cast<float>(elemCnt)}, {1}, elemTy)
          .value();
  Value elemCntRcp = rewriter.create<tosa::ReciprocalOp>(
      op.getLoc(), elemCntConst.getType(), elemCntConst);

  // Broadcast type and shape for various intermediate values.
  SmallVector<int64_t> bcastOutShape;
  for (auto en : llvm::enumerate(inputTypeShape)) {
    bcastOutShape.push_back(
        static_cast<int64_t>(en.index()) >= meanAndVarShapeRank ? 1
                                                                : en.value());
  }
  auto bcastOutType =
      RankedTensorType::get(makeShapeLLVMCompatible(bcastOutShape), elemTy);

  // Compute mean.
  Value sum = computeSumAndReshape(adaptor.getInput(), inputType, bcastOutType,
                                   bcastOutShape);
  Value meanVal = rewriter.create<tosa::MulOp>(op.getLoc(), bcastOutType, sum,
                                               elemCntRcp, /*shift=*/0);

  // Compute variance.
  Value squareSumSub = rewriter.create<tosa::SubOp>(
      op.getLoc(), inputType, adaptor.getInput(), meanVal);
  Value squareSum = rewriter.create<tosa::MulOp>(op.getLoc(), inputType,
                                                 squareSumSub, squareSumSub, 0);

  Value squareSumReduced =
      computeSumAndReshape(squareSum, inputType, bcastOutType, bcastOutShape);
  Value varianceVal = rewriter.create<tosa::MulOp>(
      op.getLoc(), bcastOutType, squareSumReduced, elemCntRcp, /*shift=*/0);

  // Reshape weight and bias.
  SmallVector<int64_t> weightAndBiasBcastShape;
  for (auto en :
       llvm::enumerate(makeShapeTorchCompatible(inputType.getShape()))) {
    weightAndBiasBcastShape.push_back(
        static_cast<int64_t>(en.index()) < meanAndVarShapeRank ? 1
                                                               : en.value());
  }
  auto weightAndMeanBcastType = RankedTensorType::get(
      makeShapeLLVMCompatible(weightAndBiasBcastShape), elemTy);

  Value weightVal = rewriter.create<tosa::ReshapeOp>(
      op.getLoc(), weightAndMeanBcastType, adaptor.getWeight(),
      rewriter.getDenseI64ArrayAttr(weightAndBiasBcastShape));

  Value biasVal = rewriter.create<tosa::ReshapeOp>(
      op.getLoc(), weightAndMeanBcastType, adaptor.getBias(),
      rewriter.getDenseI64ArrayAttr(weightAndBiasBcastShape));

  double eps;
  if (!matchPattern(op.getEps(), m_TorchConstantFloat(&eps)))
    return rewriter.notifyMatchFailure(op, "eps must be a scalar constant");
  auto epsilonConst =
      tosa::getConstTensor<float>(rewriter, op.getOperation(),
                                  {static_cast<float>(eps)}, {}, elemTy)
          .value();

  // Compute layer norm.
  auto layerNorm =
      computeBatchNorm(op, rewriter, outType, adaptor.getInput(), varianceVal,
                       epsilonConst, meanVal, weightVal, biasVal);

  rewriter.replaceOp(op, {layerNorm, meanVal, varianceVal});

  return success();
}

// Torch constants are converted to tosa.const .
template <>
LogicalResult ConvertAtenOp<ValueTensorLiteralOp>::matchAndRewrite(
    ValueTensorLiteralOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto outputTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

  // Tensors with integer types need to be converted to signless integer
  // element type. All tensors with element types other than integer can reuse
  // existing elements attribute.
  // TODO: what about unsigned integer?
  if (auto elements = dyn_cast<DenseIntElementsAttr>(op.getValueAttr())) {
    if (elements.getElementType().isSignedInteger()) {
      Type builtinTensorElemTy = outputTy.getElementType();
      unsigned bitWidth = builtinTensorElemTy.getIntOrFloatBitWidth();
      DenseElementsAttr valueAttr =
          elements.mapValues(builtinTensorElemTy, [&](const APInt &v) {
            return APInt(bitWidth, v.getSExtValue());
          });
      rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, outputTy, valueAttr);
      return success();
    }
  }
  rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, outputTy, adaptor.getValue());
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenFlattenUsingIntsOp>::matchAndRewrite(
    AtenFlattenUsingIntsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor type
  auto selfType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op,
                                       "Only ranked tensor types supported");

  int64_t selfRank = selfType.getRank();

  int64_t start_dim, end_dim;

  if (!matchPattern(op.getStartDim(), m_TorchConstantInt(&start_dim)))
    return rewriter.notifyMatchFailure(op,
                                       "start_dim must be a Scalar constant");
  start_dim = toPositiveDim(start_dim, selfRank);

  if (!matchPattern(op.getEndDim(), m_TorchConstantInt(&end_dim)))
    return rewriter.notifyMatchFailure(op, "end_dim must be a Scalar constant");
  end_dim = toPositiveDim(end_dim, selfRank);

  if (selfRank > 0 && !isValidDim(start_dim, selfRank))
    return rewriter.notifyMatchFailure(op, "start_dim is statically invalid");
  if (selfRank > 0 && !isValidDim(end_dim, selfRank))
    return rewriter.notifyMatchFailure(op, "end_dim is statically invalid");
  if (end_dim < start_dim)
    return rewriter.notifyMatchFailure(op,
                                       "end_dim must be larger than start_dim");

  SmallVector<int64_t> newShape;
  for (auto s :
       llvm::enumerate(makeShapeTorchCompatible(selfType.getShape()))) {
    int64_t idx = s.index();
    if (idx < start_dim || idx > end_dim) {
      newShape.push_back(s.value());
    } else {
      if (idx == start_dim)
        newShape.push_back(s.value());
      // Only updating when the shapes are static
      else if (s.value() != kUnknownSize && newShape.back() != kUnknownSize)
        newShape.back() *= s.value();
      else
        newShape.back() = kUnknownSize;
    }
  }

  // Handle the Scalar case
  if (newShape.size() == 0)
    newShape.push_back(1);

  auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                       selfType.getElementType());
  auto reshapeOp =
      rewriter.create<tosa::ReshapeOp>(op.getLoc(), newType, adaptor.getSelf(),
                                       rewriter.getDenseI64ArrayAttr(newShape));

  rewriter.replaceOpWithNewOp<tensor::CastOp>(
      op, getTypeConverter()->convertType(op.getType()), reshapeOp);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenUnflattenIntOp>::matchAndRewrite(
    AtenUnflattenIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor type
  auto selfType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op,
        "Only ranked tensor types with static shapes are currently supported");

  int64_t selfRank = selfType.getRank();
  int64_t dim;

  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");

  SmallVector<int64_t> sizes;
  if (!matchPattern(op.getSizes(), m_TorchListOfConstantInts(sizes)))
    return rewriter.notifyMatchFailure(
        op, "Only constant sizes are currently supported");

  if (selfRank > 0 && !isValidDim(dim, selfRank))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  SmallVector<int64_t> newShape;
  for (auto s :
       llvm::enumerate(makeShapeTorchCompatible(selfType.getShape()))) {
    int64_t idx = s.index();
    if (idx < dim || idx > dim) {
      newShape.push_back(s.value());
    } else {
      auto sum = 1;
      for (auto newDims : sizes) {
        newShape.push_back(newDims);
        sum *= newDims;
      }
      if (sum != s.value())
        return rewriter.notifyMatchFailure(op,
                                           "sizes mismatch with original dim");
    }
  }

  auto newType = RankedTensorType::get(makeShapeLLVMCompatible(newShape),
                                       selfType.getElementType());

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(newType), adaptor.getSelf(),
      rewriter.getDenseI64ArrayAttr(newShape));

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenPermuteOp>::matchAndRewrite(
    AtenPermuteOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a ranked tensor type
  auto selfType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op,
        "Only ranked tensor types with static shapes are currently supported");

  SmallVector<int64_t> dimListInt64;
  if (!matchPattern(adaptor.getDims(), m_TorchListOfConstantInts(dimListInt64)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dimensions are currently supported");
  SmallVector<int32_t> dimListInt32;
  copy(dimListInt64, std::back_inserter(dimListInt32));

  int64_t selfRank = selfType.getRank();
  // TODO: If this is already verified on the op then we can drop checking here.
  for (auto &d : dimListInt32) {
    d = toPositiveDim(d, selfRank);
    if (!isValidDim(d, selfRank))
      return rewriter.notifyMatchFailure(op, "Not all dims are valid");
  }

  auto transposeDimsConst = mlir::tosa::getConstTensor<int32_t>(
      rewriter, op.getOperation(), dimListInt32, {selfRank});

  rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      transposeDimsConst.value());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenLog2Op>::matchAndRewrite(
    AtenLog2Op op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  // Constant value of ln2.
  SmallVector<int64_t> ln2Shape(selfType.getRank(), 1);
  auto ln2Op = tosa::getConstTensor<float>(rewriter, op, {0.69314718056f},
                                           ln2Shape, selfType.getElementType())
                   .value();
  auto rcpOp =
      rewriter.create<tosa::ReciprocalOp>(op.getLoc(), ln2Op.getType(), ln2Op);

  auto outType = getTypeConverter()->convertType(op.getType());
  auto logOp =
      rewriter.create<tosa::LogOp>(op.getLoc(), outType, adaptor.getSelf());
  rewriter.replaceOpWithNewOp<tosa::MulOp>(op, outType, logOp, rcpOp,
                                           /*shift=*/0);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenThresholdOp>::matchAndRewrite(
    AtenThresholdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat())
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");

  // Integer types with width > 32 are not supported
  auto selfIntType = dyn_cast<IntegerType>(selfElemTy);
  if (selfIntType && selfIntType.getWidth() > 32) {
    return rewriter.notifyMatchFailure(
        op, "Integer types with width greater than 32 are not supported");
  }

  SmallVector<int64_t> constTypeShape(selfType.getRank(), 1);
  Value threshold, value;
  if (failed(torchScalarToTosaTensor(rewriter, op, op.getThreshold(), threshold,
                                     selfElemTy, constTypeShape)))
    return rewriter.notifyMatchFailure(
        op, "Only scalar constant is supported for threshold");

  if (failed(torchScalarToTosaTensor(rewriter, op, op.getValue(), value,
                                     selfElemTy, constTypeShape)))
    return rewriter.notifyMatchFailure(
        op, "Only scalar constant is supported for value");

  // Threshold only clamps the upper values. tosa::ClampOp has the same
  // value for both threshold and clamped value so cannot be used.
  auto outType = getTypeConverter()->convertType(op.getType());

  auto cmpOp = rewriter.create<tosa::GreaterOp>(
      op.getLoc(),
      RankedTensorType::get(selfType.getShape(), rewriter.getIntegerType(1)),
      adaptor.getSelf(), threshold);

  rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, outType, cmpOp,
                                              adaptor.getSelf(), value);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenUnsqueezeOp>::matchAndRewrite(
    AtenUnsqueezeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType) {
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");
  }

  auto selfRank = selfType.getRank();
  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");
  }

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");

  // toPositiveDim converts negative dims to the range [0, inputRank). So, -1
  // will be converted to inputRank-1. For `torch.unsqueeze` op, -1 has to be
  // converted to inputRank, and the valid dim range is [0, inputRank + 1).
  dim = toPositiveDim(dim, selfRank + 1);
  if (!isValidDim(dim, selfRank + 1))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  SmallVector<int64_t> outShape;
  for (auto en :
       llvm::enumerate(makeShapeTorchCompatible(selfType.getShape()))) {
    if (static_cast<int64_t>(en.index()) == dim)
      outShape.push_back(1);

    outShape.push_back(en.value());
  }
  if (dim == selfRank)
    outShape.push_back(1);

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      rewriter.getDenseI64ArrayAttr(outShape));

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenContiguousOp>::matchAndRewrite(
    AtenContiguousOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  // FIXME: memory_format is not handled.

  rewriter.replaceOp(op, adaptor.getSelf());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenDropoutOp>::matchAndRewrite(
    AtenDropoutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getInput().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  // FIXME: train and p are not handled.

  bool train;
  if (!matchPattern(op.getTrain(), m_TorchConstantBool(&train)))
    return rewriter.notifyMatchFailure(op, "train must be a Scalar constant");

  if (train)
    return rewriter.notifyMatchFailure(op, "train must be false");

  rewriter.replaceOpWithNewOp<tosa::CastOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getInput());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenViewOp>::matchAndRewrite(
    AtenViewOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");
  }

  SmallVector<int64_t> outShape;
  if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(outShape)))
    return rewriter.notifyMatchFailure(op,
                                       "size must consist of Scalar constants");

  // the shape -1 is inferred from other dimensions
  size_t countNegativeShape{0};
  // Check at most one -1 shape
  for (size_t i = 0; i < outShape.size(); i++) {
    if (outShape[i] < 0) {
      countNegativeShape++;
      if (countNegativeShape > 1)
        return rewriter.notifyMatchFailure(op, "At most one -1 shape");
    }
  }

  auto inputShape = selfType.getShape();
  size_t totalSize = 1;
  for (size_t i = 0; i < inputShape.size(); i++) {
    totalSize *= inputShape[i];
  }

  size_t otherSize = 1;
  for (size_t i = 0; i < outShape.size(); i++) {
    if (outShape[i] > 0) {
      otherSize *= outShape[i];
    }
  }
  for (size_t i = 0; i < outShape.size(); i++) {
    if (outShape[i] < 0) {
      outShape[i] = totalSize / otherSize;
      break;
    }
  }

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      rewriter.getDenseI64ArrayAttr(outShape));

  return success();
}

static Value approximateErfOp(ConversionPatternRewriter &rewriter,
                              Operation *op, Value x, Type dtype) {
  // Using:
  // https://en.wikipedia.org/wiki/Error_function#Numerical_approximations with
  // maximum error as 5 x 10^-4 where a1 = 0.278393, a2 = 0.230389, a3 =
  // 0.000972, a4 = 0.078108.
  //
  // Erf = 1 - 1 / (1 + a1X + a2X + a3X + a4X)^4

  auto outType = cast<TensorType>(x.getType());
  auto loc = op->getLoc();
  auto absX = rewriter.create<tosa::AbsOp>(loc, outType, x);
  auto zero = tosa::getConstTensor<float>(rewriter, op, 0, {}, dtype).value();
  auto one = tosa::getConstTensor<float>(rewriter, op, 1, {}, dtype).value();

  auto a1 =
      tosa::getConstTensor<float>(rewriter, op, 0.278393f, {}, dtype).value();
  auto a1X = rewriter.create<tosa::MulOp>(loc, outType, a1, absX, /*shift=*/0);
  auto sum = rewriter.create<tosa::AddOp>(loc, outType, a1X, one);

  auto a2 =
      tosa::getConstTensor<float>(rewriter, op, 0.230389f, {}, dtype).value();
  auto x2 = rewriter.create<tosa::MulOp>(loc, outType, absX, absX, /*shift=*/0);
  auto a2X = rewriter.create<tosa::MulOp>(loc, outType, a2, x2, /*shift=*/0);
  sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a2X);

  auto a3 =
      tosa::getConstTensor<float>(rewriter, op, 0.000972f, {}, dtype).value();
  auto x3 = rewriter.create<tosa::MulOp>(loc, outType, x2, absX, /*shift=*/0);
  auto a3X = rewriter.create<tosa::MulOp>(loc, outType, a3, x3, /*shift=*/0);
  sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a3X);

  auto a4 =
      tosa::getConstTensor<float>(rewriter, op, 0.078108f, {}, dtype).value();
  auto x4 = rewriter.create<tosa::MulOp>(loc, outType, x3, absX, /*shift=*/0);
  auto a4X = rewriter.create<tosa::MulOp>(loc, outType, a4, x4, /*shift=*/0);
  sum = rewriter.create<tosa::AddOp>(loc, outType, sum, a4X);

  auto rcprl = rewriter.create<tosa::ReciprocalOp>(loc, outType, sum);
  auto rcprl2 =
      rewriter.create<tosa::MulOp>(loc, outType, rcprl, rcprl, /*shift=*/0);
  auto rcprl4 =
      rewriter.create<tosa::MulOp>(loc, outType, rcprl2, rcprl2, /*shift=*/0);
  auto erf = rewriter.create<tosa::SubOp>(loc, outType, one, rcprl4);

  // Deal with negative x.
  auto cond = rewriter.create<tosa::GreaterEqualOp>(
      loc,
      RankedTensorType::get(outType.getShape(), rewriter.getIntegerType(1)), x,
      zero);
  auto negateErf = rewriter.create<tosa::NegateOp>(loc, outType, erf);

  return rewriter.create<tosa::SelectOp>(loc, outType, cond, erf, negateErf);
}

static Value buildUnitNormalCdf(ConversionPatternRewriter &rewriter,
                                Operation *op, Value x, Type dtype) {
  auto zero = tosa::getConstTensor<float>(rewriter, op, 0, {}, dtype).value();
  auto one = tosa::getConstTensor<float>(rewriter, op, 1, {}, dtype).value();

  auto loc = op->getLoc();

  // buildNormalCdf, mean = zero, sigma = one
  auto outType = x.getType();
  auto mean = zero;
  Value xMinusMean = rewriter.create<tosa::SubOp>(loc, outType, x, mean);
  // rsqrt of 2
  Value rsqrt2 =
      tosa::getConstTensor<float>(rewriter, op, 0.70710678f, {}, dtype).value();

  Value erfArg = rewriter.create<tosa::MulOp>(loc, outType, xMinusMean, rsqrt2,
                                              /*shift=*/0);
  Value erf = approximateErfOp(rewriter, op, erfArg, dtype);
  Value erfPlus1 = rewriter.create<tosa::AddOp>(loc, outType, one, erf);
  Value oneHalf =
      tosa::getConstTensor<float>(rewriter, op, 0.5, {}, dtype).value();

  Value normalCdf = rewriter.create<tosa::MulOp>(loc, outType, oneHalf,
                                                 erfPlus1, /*shift=*/0);
  return normalCdf;
}

// This lowering is based on Torch to LinAlg lowering.
template <>
LogicalResult ConvertAtenOp<AtenGeluOp>::matchAndRewrite(
    AtenGeluOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!isa<mlir::FloatType>(selfElemTy)) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization supported");
  }

  // TODO: Handle approximate.
  std::string approximate;
  if (!matchPattern(op.getApproximate(), m_TorchConstantStr(approximate)) ||
      approximate != "none") {
    return rewriter.notifyMatchFailure(op, "Unsupported value of approximate");
  }

  Value cdf = buildUnitNormalCdf(rewriter, op, adaptor.getSelf(), selfElemTy);
  cdf = rewriter.createOrFold<tosa::CastOp>(
      op->getLoc(),
      cast<RankedTensorType>(cdf.getType()).cloneWith({}, selfElemTy), cdf);

  rewriter.replaceOpWithNewOp<tosa::MulOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(), cdf,
      /*shift=*/0);

  return success();
}

// This lowering is based on Torch to LinAlg lowering.
template <>
LogicalResult ConvertAtenOp<AtenGeluBackwardOp>::matchAndRewrite(
    AtenGeluBackwardOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");

  auto selfElemTy = selfType.getElementType();
  if (!isa<mlir::FloatType>(selfElemTy)) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point datatype legalization supported");
  }

  // TODO: Handle approximate.
  std::string approximate;
  if (!matchPattern(op.getApproximate(), m_TorchConstantStr(approximate)) ||
      approximate != "none") {
    return rewriter.notifyMatchFailure(op, "Unsupported value of approximate");
  }

  auto loc = op->getLoc();

  const float cstAlpha0 = 1.12837916709551257390f;
  const float cstAlpha1 = 0.70710678118654752440f;
  const float oneHalf = 0.5f;
  const float kAlpha = cstAlpha0 * cstAlpha1;

  Value kAlphaHalf = tosa::getConstTensor<float>(rewriter, op, kAlpha * oneHalf,
                                                 {}, selfElemTy)
                         .value();
  Value negOneHalf =
      tosa::getConstTensor<float>(rewriter, op, -0.5f, {}, selfElemTy).value();
  Value inputSquared = rewriter.create<tosa::MulOp>(
      loc, selfType, adaptor.getSelf(), adaptor.getSelf(), /*shift=*/0);
  Value negHalfInputSquared = rewriter.create<tosa::MulOp>(
      loc, selfType, inputSquared, negOneHalf, /*shift=*/0);
  Value dinput =
      rewriter.create<tosa::ExpOp>(loc, selfType, negHalfInputSquared);
  Value cdf = buildUnitNormalCdf(rewriter, op, adaptor.getSelf(), selfElemTy);
  Value dinputInput = rewriter.create<tosa::MulOp>(
      loc, selfType, dinput, adaptor.getSelf(), /*shift=*/0);
  Value dinputInputAlpha = rewriter.create<tosa::MulOp>(
      loc, selfType, dinputInput, kAlphaHalf, /*shift=*/0);
  Value cdfExt =
      rewriter.create<tosa::AddOp>(loc, selfType, dinputInputAlpha, cdf);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(
      op, getTypeConverter()->convertType(op.getType()),
      adaptor.getGradOutput(), cdfExt,
      /*shift=*/0);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenHardtanhBackwardOp>::matchAndRewrite(
    AtenHardtanhBackwardOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType) {
    return rewriter.notifyMatchFailure(
        op, "Only tensor types are currently supported");
  }

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");
  }

  // Integer types with width > 32 are not supported
  auto selfIntType = dyn_cast<IntegerType>(selfElemTy);
  if (selfIntType && selfIntType.getWidth() > 32) {
    return rewriter.notifyMatchFailure(
        op, "Integer types with width greater than 32 are not supported");
  }

  Value gradOutput = adaptor.getGradOutput();
  auto gradOutputType = dyn_cast<TensorType>(adaptor.getSelf().getType());

  Type gradOutputElemType = gradOutputType.getElementType();

  if (selfElemTy != gradOutputElemType) {
    return rewriter.notifyMatchFailure(
        op,
        "Input element type should be same as the grad_output element type.");
  }

  SmallVector<int64_t> constTypeShape(selfType.getRank(), 1);
  Value maxVal, minVal;

  if (failed(torchScalarToTosaTensor(rewriter, op, op.getMinVal(), minVal,
                                     selfElemTy, constTypeShape))) {
    return rewriter.notifyMatchFailure(op, "Only scalar constant is supported");
  }

  if (failed(torchScalarToTosaTensor(rewriter, op, op.getMaxVal(), maxVal,
                                     selfElemTy, constTypeShape))) {
    return rewriter.notifyMatchFailure(op, "Only scalar constant is supported");
  }

  Value replace =
      tosa::getConstTensor<float>(rewriter, op, 0, {}, selfElemTy).value();
  Type outType = getTypeConverter()->convertType(op.getType());

  Value lesser = rewriter.create<tosa::GreaterOp>(
      op.getLoc(),
      RankedTensorType::get(selfType.getShape(), rewriter.getIntegerType(1)),
      minVal, adaptor.getSelf());

  Value greater = rewriter.create<tosa::GreaterOp>(
      op.getLoc(),
      RankedTensorType::get(selfType.getShape(), rewriter.getIntegerType(1)),
      adaptor.getSelf(), maxVal);

  Value cmp = rewriter.create<tosa::LogicalOrOp>(
      op.getLoc(),
      RankedTensorType::get(selfType.getShape(), rewriter.getIntegerType(1)),
      lesser, greater);

  rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, outType, cmp, replace,
                                              gradOutput);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenEmbeddingOp>::matchAndRewrite(
    AtenEmbeddingOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value weight = adaptor.getWeight();
  Value indices = adaptor.getIndices();
  RankedTensorType outType =
      cast<RankedTensorType>(typeConverter->convertType(op.getType()));

  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!indicesType || !isa<IntegerType>(indicesType.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "Indices must be of integer tensor type");

  auto weightType = cast<RankedTensorType>(weight.getType());
  if (weightType.getRank() != 2)
    return op.emitError("weight must be of rank 2");

  // FIXME: padding_idx, scale_grad_by_freq and sparse are not handled yet.
  int64_t paddingIdx;
  if (!matchPattern(op.getPaddingIdx(), m_TorchConstantInt(&paddingIdx)))
    return rewriter.notifyMatchFailure(
        op, "only supports constant int padding_idx for embedding op");

  bool scaleGradByFreq;
  if (!matchPattern(op.getScaleGradByFreq(),
                    m_TorchConstantBool(&scaleGradByFreq)))
    return rewriter.notifyMatchFailure(
        op, "only supports constant bool scale_grad_by_freq for embedding op");
  if (scaleGradByFreq)
    return rewriter.notifyMatchFailure(
        op,
        "only supports scale_grad_by_freq equals to False for embedding op");

  bool isSparse;
  if (!matchPattern(op.getSparse(), m_TorchConstantBool(&isSparse)))
    return rewriter.notifyMatchFailure(
        op, "only supports constant bool sparse for embedding op");
  if (isSparse)
    return rewriter.notifyMatchFailure(
        op, "only support sparse equals to False for embedding op");

  // For inference:
  //    Weights [num_embeddings, embedding_dim], Indices [X, Y]
  //    Output [X, Y, embedding_dim] = Weights[Indices[x, y]] forall x in X, y
  //    in Y
  //
  //    Condition: num_embeddings > Indices [x, y] forall x in X, y in Y

  // Reshape the weight, since tosa.gather expects a 3D tensor
  auto indicesShape = makeShapeTorchCompatible(indicesType.getShape());
  auto weightShape = makeShapeTorchCompatible(weightType.getShape());

  SmallVector<int64_t> newWeightShape = {1};
  for (auto s : weightShape)
    newWeightShape.push_back(s);

  auto reshapedWeight = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(newWeightShape),
                            weightType.getElementType()),
      weight, rewriter.getDenseI64ArrayAttr(newWeightShape));

  int64_t numIndices = 1;
  if (indicesType.hasStaticShape()) {
    for (auto s : indicesShape)
      numIndices *= s;
  } else {
    numIndices = kUnknownSize;
  }

  SmallVector<int64_t> newIndicesShape = {1, numIndices};
  auto reshapedIndices = rewriter.create<tosa::ReshapeOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(newIndicesShape),
                            indicesType.getElementType()),
      indices, rewriter.getDenseI64ArrayAttr(newIndicesShape));

  auto castIndices = rewriter.create<tosa::CastOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(newIndicesShape),
                            rewriter.getIntegerType(32)),
      reshapedIndices);

  SmallVector<int64_t> intermediateOutShape = {1, numIndices, weightShape[1]};
  auto gatherOp = rewriter.create<tosa::GatherOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(intermediateOutShape),
                            weightType.getElementType()),
      reshapedWeight, castIndices);

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
      op, outType, gatherOp,
      rewriter.getDenseI64ArrayAttr(
          makeShapeTorchCompatible(outType.getShape())));

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenTransposeIntOp>::matchAndRewrite(
    AtenTransposeIntOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  // Only statically resolvable values are currently supported
  int64_t dim0, dim1;
  if (!matchPattern(op.getDim0(), m_TorchConstantInt(&dim0)))
    return rewriter.notifyMatchFailure(op, "dim0 must be a Scalar constant");

  if (!matchPattern(op.getDim1(), m_TorchConstantInt(&dim1)))
    return rewriter.notifyMatchFailure(op, "dim1 must be a Scalar constant");

  dim0 = toPositiveDim(dim0, selfType.getRank());
  dim1 = toPositiveDim(dim1, selfType.getRank());

  auto selfRank = selfType.getRank();
  if (!isValidDim(dim0, selfRank) || !isValidDim(dim1, selfRank))
    return rewriter.notifyMatchFailure(
        op, "dim0 and dim1 must be less than tensor rank");

  SmallVector<int32_t> transposeDims;
  for (auto i = 0; i < selfType.getRank(); ++i)
    transposeDims.push_back(i);

  transposeDims[dim0] = dim1;
  transposeDims[dim1] = dim0;

  auto transposeDimsConst = mlir::tosa::getConstTensor<int32_t>(
      rewriter, op.getOperation(), transposeDims, {selfType.getRank()});

  rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.getSelf(),
      transposeDimsConst.value());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenMaxDimOp>::matchAndRewrite(
    AtenMaxDimOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto indicesType =
      dyn_cast<TensorType>(getTypeConverter()->convertType(op.getType(1)));
  if (!indicesType)
    return rewriter.notifyMatchFailure(op, "Only tensor types are supported");

  auto selfElemType = selfType.getElementType();
  auto indicesElemType = indicesType.getElementType();

  // Only statically deducible values are currently supported
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");

  dim = toPositiveDim(dim, selfType.getRank());

  if (!isValidDim(dim, selfType.getRank()))
    return rewriter.notifyMatchFailure(op, "dim must be less than tensor rank");

  bool keepDim;
  if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim)))
    return rewriter.notifyMatchFailure(op, "keepdim must be a Scalar constant");

  SmallVector<int64_t> reducedShape, prunedShape;
  for (auto en :
       llvm::enumerate(makeShapeTorchCompatible(selfType.getShape()))) {
    if (static_cast<int64_t>(en.index()) == dim) {
      reducedShape.push_back(1);
      continue;
    }
    reducedShape.push_back(en.value());
    prunedShape.push_back(en.value());
  }

  auto dimAttr = rewriter.getIntegerAttr(rewriter.getI32Type(), dim);
  auto prunedShapeAttr = rewriter.getDenseI64ArrayAttr(prunedShape);

  Value reduceMax = rewriter.create<tosa::ReduceMaxOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(reducedShape),
                            selfElemType),
      adaptor.getSelf(), dimAttr);

  Value argMax = rewriter.create<tosa::ArgMaxOp>(
      op->getLoc(),
      RankedTensorType::get(makeShapeLLVMCompatible(prunedShape),
                            indicesElemType),
      adaptor.getSelf(), dimAttr);

  if (argMax.getType() != indicesType) {
    argMax = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(), indicesType, argMax,
        rewriter.getDenseI64ArrayAttr(reducedShape));
  }

  if (!keepDim) {
    reduceMax = rewriter.create<tosa::ReshapeOp>(
        op->getLoc(),
        RankedTensorType::get(makeShapeLLVMCompatible(prunedShape),
                              selfElemType),
        reduceMax, prunedShapeAttr);
  }

  rewriter.replaceOp(op, {reduceMax, argMax});

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSliceTensorOp>::matchAndRewrite(
    AtenSliceTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  if (op->use_empty()) {
    rewriter.eraseOp(op);
    return success();
  }

  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  auto outTy =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!outTy) {
    return rewriter.notifyMatchFailure(op, "output type must be ranked");
  }
  if (outTy.hasStaticShape() && outTy.getNumElements() == 0) {
    return rewriter.notifyMatchFailure(op,
                                       "tosa.slice does not support zero size");
  }

  // Only statically deducible values are currently supported
  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(op, "dim must be a Scalar constant");

  dim = toPositiveDim(dim, selfType.getRank());

  if (!isValidDim(dim, selfType.getRank()))
    return rewriter.notifyMatchFailure(op, "dim must less than tensor rank");

  auto sizeOfDim = selfType.getDimSize(dim);

  int64_t start;
  if (!matchPattern(op.getStart(), m_TorchConstantInt(&start)))
    return rewriter.notifyMatchFailure(op, "start must be a Scalar constant");

  start = toPositiveDim(start, selfType.getShape()[dim]);
  start = std::clamp(start, (int64_t)0, sizeOfDim);

  int64_t end;
  if (!matchPattern(op.getEnd(), m_TorchConstantInt(&end))) {
    if (isa<ConstantNoneOp>(op.getEnd().getDefiningOp()))
      end = selfType.getShape()[dim];
    else
      return rewriter.notifyMatchFailure(op, "end must be a Scalar constant");
  }
  // support for end < 0
  end = toPositiveDim(end, selfType.getShape()[dim]);
  // support for end out of upper bound
  end = (end > selfType.getShape()[dim] ? selfType.getShape()[dim] : end);
  // Handle start > end
  end = std::clamp(end, (int64_t)0, sizeOfDim);

  int64_t step;
  if (!matchPattern(op.getStep(), m_TorchConstantInt(&step)))
    return rewriter.notifyMatchFailure(op, "step must be a Scalar constant");

  if (sizeOfDim % step != 0) {
    return rewriter.notifyMatchFailure(op, "size must be divisible by step");
  }

  // We handle step by splitting the dimension dim into two dimensions,
  // where the second one has size 'step'.
  // E.g. to take slice with step 3 out of dim=0 of [6, 10], we first
  // reshape into [2, 3, 10].
  SmallVector<int64_t> newShape{selfType.getShape()};
  newShape[dim] /= step;
  newShape.insert(newShape.begin() + dim + 1, step);

  auto reshaped =
      tosa::reshapeTo(op->getLoc(), rewriter, adaptor.getSelf(), newShape);

  SmallVector<int64_t> startSlice(reshaped.getType().getRank(), 0);

  startSlice[dim] = start / step;
  startSlice[dim + 1] = start % step;

  SmallVector<int64_t> sliceShape{outTy.getShape()};
  sliceShape.insert(sliceShape.begin() + dim + 1, 1);

  auto slice = rewriter.create<tosa::SliceOp>(
      op.getLoc(), outTy.cloneWith(sliceShape, outTy.getElementType()),
      reshaped, rewriter.getDenseI64ArrayAttr(startSlice),
      rewriter.getDenseI64ArrayAttr(sliceShape));

  auto out = tosa::reshapeTo(op->getLoc(), rewriter, slice, outTy.getShape());

  rewriter.replaceOp(op, out);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenBroadcastToOp>::matchAndRewrite(
    AtenBroadcastToOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  auto selfElemTy = selfType.getElementType();
  if (!selfElemTy.isIntOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");
  }

  SmallVector<int64_t> resultShape;
  if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(resultShape)))
    return rewriter.notifyMatchFailure(op,
                                       "size must consist of Scalar constants");
  // Get the result type
  auto resultType = getTypeConverter()->convertType(op.getType());

  int64_t numBroadcastedDims = resultShape.size() - selfType.getRank();
  assert(numBroadcastedDims >= 0 &&
         "numBroadcastedDims must be positive or zero.");

  // Result dimension -1 means not changing the size of that dimension.
  // Adjust it by assigning its inputShape according to the rank difference
  // between input and result.
  SmallVector<int64_t> inputShape(
      makeShapeTorchCompatible(selfType.getShape()));
  for (auto shape : llvm::enumerate(inputShape)) {
    auto index = shape.index() + numBroadcastedDims;
    if (resultShape[index] == -1)
      resultShape[index] = shape.value();
  }

  // If there are still unknown dimensions, nothing can be done.
  if (llvm::any_of(resultShape, [&](auto dim) { return dim == -1; })) {
    return rewriter.notifyMatchFailure(
        op, "cannot propagate unknown (-1) dimension "
            "as it is not presented in the input.");
  }

  // Add 1 to each broadcasted dimension in the input.
  // Broadcasted dimensions are the outermost ones.
  SmallVector<int64_t> broadcastedDims(numBroadcastedDims, 1);
  inputShape.insert(inputShape.begin(), broadcastedDims.begin(),
                    broadcastedDims.end());

  // Check for identity case i.e, for ex: [a, b, c] -> [a, b, c]. If this is
  // true then we can replace the op result with the input operand directly.
  if (llvm::equal(inputShape, resultShape) && !numBroadcastedDims) {
    // If we reach here, then it means that the broadcasting is not required
    // since the input and result are of same shape.
    op.replaceAllUsesWith(op.getSelf());
    rewriter.eraseOp(op);
    return success();
  } else if (selfType.hasRank() && (inputShape.size() == resultShape.size() ||
                                    selfType.getRank() == 0)) {
    // Right now to support limited cases where input and result shape are not
    // equal, we can put a constraint that either the input should be of rank
    // 0 or the rank of input tensor and result should be equal. And then we
    // can check for broadcasting compatibility for the latter case. For
    // broadcasting compatibility, either the shape of input and result should
    // be equal at each dimenion or one of them should be 1.
    if (selfType.getRank() != 0) {
      for (unsigned i = 0; i < inputShape.size(); i++) {
        if (inputShape[i] != resultShape[i] && inputShape[i] != 1 &&
            resultShape[i] != 1) {
          return rewriter.notifyMatchFailure(
              op, "unimplemented: either the shape of input and result should "
                  "be equal at each dimension or one of them should be 1.");
        }
      }
    }

    // If the above condition hold true then we can directly create a const
    // zero tensor of shape same as the result shape.
    SmallVector<int64_t> zeroTensorShape{resultShape};

    // create the 0 constant tensor
    int64_t totalNumElements = 1;
    for (auto dimSize : zeroTensorShape) {
      totalNumElements = dimSize * totalNumElements;
    }
    // There is some danger here. For edge cases in floating point, x + 0 != x.
    // The cases are denormalized values, which may get flushed, and -0 + 0 =
    // +0. (sign bit flips). These are probably acceptable in the short term,
    // but we should put a comment acknowledging the danger, as there isn't an
    // op that avoids the denorm flushing.
    Value zeroTensor =
        tosa::getZerosLikeTensor(rewriter, op, resultType).value();

    // Use add broadcast
    auto newOp = rewriter.createOrFold<tosa::AddOp>(
        op.getLoc(), resultType, adaptor.getSelf(), zeroTensor);
    rewriter.replaceOp(op, newOp);
    return success();
  }
  return rewriter.notifyMatchFailure(
      op,
      "unimplemented: broadcasts other than same rank or zero ranked tensor.");
}

template <>
LogicalResult ConvertAtenOp<AtenGatherOp>::matchAndRewrite(
    AtenGatherOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // For easy understanding of this algorithm, I will comment the code with an
  // exact example: torch.aten.gather (!torch.vtensor<[1,4,3],f32>,
  // !torch.int-1, !torch.vtensor<[1,4,2],si64>)
  // -> !torch.vtensor<[1,4,2],f32>
  // https://gist.github.com/AmosLewis/2f18434397025211da4491735bcc6db6

  // Not a tensor type.
  auto input = adaptor.getSelf();
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  if (!inputType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType input are currently supported");

  auto index = adaptor.getIndex();
  auto indexType = dyn_cast<RankedTensorType>(adaptor.getIndex().getType());
  auto inputShape = inputType.getShape();
  int paramsRank = inputShape.size();

  if (!indexType)
    return rewriter.notifyMatchFailure(
        op, "Only RankedTensorType index are currently supported");

  // Check `index` and `input` param should have the same rank
  if (indexType.getRank() != inputType.getRank())
    return rewriter.notifyMatchFailure(
        op, "`index` and `input` param should have the same rank");

  // Dynamic shape check
  if (!inputType.hasStaticShape() || !indexType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "AtenGatherOp: support for dynamic input "
            "shape not implemented");

  // index i64 to i32 for tosa compatitable
  if (indexType.getElementType() != rewriter.getIntegerType(32)) {
    index = rewriter.create<tosa::CastOp>(
        op->getLoc(),
        RankedTensorType::get(indexType.getShape(),
                              rewriter.getIntegerType(32)),
        index);
  }

  // Get positive dim
  int64_t dim{0};
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "unimplemented: value `dim` should be a torch constant int");
  dim = toPositiveDim(dim, paramsRank);
  if (!isValidDim(dim, paramsRank))
    return rewriter.notifyMatchFailure(op, "Not dim are invalid");

  // check sparseGrad is bool type
  bool sparseGrad = false;
  if (!matchPattern(op.getSparseGrad(), m_TorchConstantBool(&sparseGrad)))
    return rewriter.notifyMatchFailure(
        op, "only constant boolean `sparse_grad` param supported");
  if (sparseGrad)
    return rewriter.notifyMatchFailure(
        op, "only constant boolean `sparse_grad` == false supported");

  // Get the output type
  auto outType = getTypeConverter()->convertType(op.getType());

  // convert torch style index and dim into tf style indices
  // tensor<[1,4,2],si64> -> tensor<[1,4,2,3],si64>
  auto indicesTf =
      tosa::convertTorchIndexToTfIndices(rewriter, op, input, index, dim);
  if (!indicesTf) {
    return rewriter.notifyMatchFailure(op,
                                       "Convert TorchIndex To TfIndices fail.");
  }

  // do the tf gathernp algorithm with tf style indices as input.
  auto result =
      tosa::convertGatherNdOp(rewriter, op, outType, input, indicesTf.value());

  if (!result) {
    return rewriter.notifyMatchFailure(op, "Convert GatherNdOp fail.");
  }
  rewriter.replaceOp(op, {result.value()});
  return success();
}

// Turn a torch.aten._index_put_impl where some entries in the indices list are
// none into multiple _index_put_impl across all elements of that dimension.
//
//  Example:
//    a = torch.aten._index_put_impl(in, [idx0, None, idx1], values)
//  where in is a 7x3x5 tensor, is equivalent to
//    tmp = torch.aten._index_put_impl(in, [idx0, [0], idx1], values)
//    tmp2 = torch.aten._index_put_impl(tmp, [idx0, [1], idx1], values)
//    a = torch.aten._index_put_impl(tmp2, [idx0, [2], idx1], values)
class SimplifyAten_IndexPutImplOpNone
    : public OpRewritePattern<Aten_IndexPutImplOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Aten_IndexPutImplOp op,
                                PatternRewriter &rewriter) const override {

    auto outTy = dyn_cast<BaseTensorType>(op.getType());
    if (!outTy || !outTy.areAllSizesKnown())
      return failure();

    SmallVector<Value> indices;
    if (!getListConstructElements(op.getIndices(), indices))
      return failure();

    for (size_t i = 0; i < indices.size(); ++i) {
      if (isa<Torch::NoneType>(indices[i].getType())) {
        Value newIndexPut = op.getSelf();
        auto si64Type =
            IntegerType::get(rewriter.getContext(), 64, IntegerType::Signed);
        Type indexType =
            ValueTensorType::get(rewriter.getContext(), {{}}, si64Type);
        for (int64_t d = 0; d < outTy.getSizes()[i]; ++d) {
          SmallVector<Value> newIndices = indices;

          newIndices[i] = rewriter.create<PrimNumToTensorScalarOp>(
              op.getLoc(), indexType,
              rewriter.create<Torch::ConstantIntOp>(op->getLoc(), d));

          Value newIndicesList = rewriter.create<PrimListConstructOp>(
              op->getLoc(), op.getIndices().getType(), newIndices);

          newIndexPut = rewriter.create<Aten_IndexPutImplOp>(
              op.getLoc(), op.getType(), newIndexPut, newIndicesList,
              op.getValues(), op.getAccumulate(), op.getUnsafe());
        }
        rewriter.replaceOp(op, newIndexPut);
        return success();
      }
    }
    return failure();
  }
};

// Turn a torch.aten._index_put_impl on a 2d [1, n] tensor into a
// torch.aten._index_put_impl on a 1d [n] tensor.
class SimplifyAten_IndexPutImplOp
    : public OpRewritePattern<Aten_IndexPutImplOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Aten_IndexPutImplOp op,
                                PatternRewriter &rewriter) const override {

    auto ty = dyn_cast<BaseTensorType>(op.getType());
    if (!ty || !ty.areAllSizesKnown()) {
      return rewriter.notifyMatchFailure(op, "Required ranked tensor type");
    }

    auto shape = ty.getSizes();
    if (shape.size() != 2 || shape[0] != 1) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: non-2d output with leading dimension of size 1");
    }
    int64_t numSelfElements = shape[1];

    auto valuesTy = dyn_cast<BaseTensorType>(op.getValues().getType());
    if (!valuesTy || !valuesTy.areAllSizesKnown()) {
      return rewriter.notifyMatchFailure(
          op, "Required ranked tensor type for values");
    }

    auto valuesShape = valuesTy.getSizes();
    if (valuesShape.size() > 2) {
      return rewriter.notifyMatchFailure(op,
                                         "unimplemented: nd values with n>=2");
    }
    if (valuesShape.size() == 2 && valuesShape[0] != 1) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: 2d values with leading dimension of size 1");
    }
    auto numValues = valuesShape.empty() ? 1 : valuesShape.back();

    SmallVector<Value> indicesList;
    if (!getListConstructElements(op.getIndices(), indicesList)) {
      return op.emitError(
          "unimplemented: the indices list is not from list construct");
    }
    // There is one indices tensor for each dimension of self.
    // Here, we know that self is 1xN, so we are only interested for the indices
    // of the 2nd dimension.
    auto indices = indicesList[1];
    auto indicesTy = dyn_cast<BaseTensorType>(indices.getType());
    if (!indicesTy || !indicesTy.areAllSizesKnown()) {
      return rewriter.notifyMatchFailure(
          op, "Required ranked tensor type for indices");
    }
    if (indicesTy.getSizes().size() > 1) {
      return rewriter.notifyMatchFailure(
          op, "Required 0d or 1d tensor for indices");
    }
    auto numIndices =
        indicesTy.getSizes().empty() ? 1 : indicesTy.getSizes()[0];

    if (indicesTy.getSizes().empty()) {
      indices = reshapeTo(op.getLoc(), rewriter, indices, {1});
    }

    // Broadcast so that values and indices have the same size
    if (numIndices == 1 && numValues > numIndices) {
      indices = broadcastTo(op.getLoc(), rewriter, indices, {numValues});
    }

    Value newIndicesList = rewriter.create<PrimListConstructOp>(
        op->getLoc(), op.getIndices().getType(), SmallVector<Value>{indices});

    auto reshapedSelf =
        reshapeTo(op.getLoc(), rewriter, op.getSelf(), {numSelfElements});

    auto values = reshapeTo(op.getLoc(), rewriter, op.getValues(), {numValues});

    // Broadcast so that values and indices have the same size
    if (numValues == 1 && numIndices > numValues) {
      values = broadcastTo(op.getLoc(), rewriter, values, {numIndices});
    }

    auto put = rewriter.create<Aten_IndexPutImplOp>(
        op.getLoc(), reshapedSelf.getType(), reshapedSelf, newIndicesList,
        values, op.getAccumulate(), op.getUnsafe());

    rewriter.replaceOp(op, reshapeTo(op.getLoc(), rewriter, put, shape));

    return success();
  }
};

// Handle Aten_IndexPutImplOp on 1d tensors
template <>
LogicalResult ConvertAtenOp<AtenIndexPutHackedTwinOp>::matchAndRewrite(
    AtenIndexPutHackedTwinOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // TOSA scatter:
  // // Copy the values_in tensor to the values_out tensor.
  // // Values not written by the scatter operation are unchanged in the output.
  // for_each(0 <= n < N, 0 <= k < K, 0 <= c < C) {
  //     value_t value = tensor_read<value_t>(values_in, [N,K,C], [n,k,c]);
  //     tensor_write<value_t>(values_out, [N,K,C], [n, k, c], value);
  // }
  // // Now perform the SCATTER operation, modifying the positions from the
  // indices tensor for_each(0 <= n < N, 0 <= w < W, 0 <= c < C) {
  //     index_t k = tensor_read<index_t>(indices, [N,W], [n,w]);
  //     REQUIRE(0 <= k && k < K);
  //     value_t value = tensor_read<value_t>(input, [N,W,C], [n,w,c]);
  //     tensor_write<value_t>(values_out, [N,K,C], [n, k, c], value);
  //     output_modified[n,k,c] = true;
  // }

  auto loc = op.getLoc();

  // Not a tensor type.
  auto self = dyn_cast<TypedValue<RankedTensorType>>(adaptor.getSelf());
  if (!self)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  if (self.getType().getRank() != 1) {
    return rewriter.notifyMatchFailure(
        op, "Only 1d input tensor are currently supported");
  }

  auto values = dyn_cast<TypedValue<RankedTensorType>>(adaptor.getValues());
  if (!values)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  // Deal with torch.prim.ListConstruct of non const value to get the index
  SmallVector<Value> indicesTorchType;
  if (!getListConstructElements(op.getIndices(), indicesTorchType))
    return op.emitError(
        "unimplemented: the tensor list is not from list construct");

  // Convert indicesTorchType to TOSA types
  auto indexTensors = getTypeConvertedValues(
      rewriter, op->getLoc(), getTypeConverter(), indicesTorchType);

  // the number of tensors in indexTensors is equal to the rank of outType
  if (indexTensors.size() != 1) {
    return rewriter.notifyMatchFailure(op, "Expected 1 indices ");
  }

  auto indices0 = indexTensors[0];
  auto indicesTy = dyn_cast<RankedTensorType>(indices0.getType());

  if (!indicesTy || indicesTy.getShape() != values.getType().getShape())
    return rewriter.notifyMatchFailure(
        op, "Expected indices to have same shape as values");

  auto outType =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!outType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  auto numInElements = self.getType().getShape()[0];
  auto numValues = values.getType().getShape()[0];

  // TOSA scatter requires 3d in and 2d indices & values
  SmallVector<int64_t> scatterInOutShape{1, numInElements, 1};
  SmallVector<int64_t> scatterIndicesShape{1, numValues};
  SmallVector<int64_t> scatterInputShape{1, numValues, 1};

  auto in = mlir::tosa::reshapeTo(loc, rewriter, self, scatterInOutShape);
  auto indices =
      mlir::tosa::reshapeTo(loc, rewriter, indices0, scatterIndicesShape);
  auto input = mlir::tosa::reshapeTo(loc, rewriter, values, scatterInputShape);

  // TOSA scatter requires 32 bit indices
  // TODO: This might break on large (sparse?) tensors that require 64 bit
  // indices
  auto indices32Ty = RankedTensorType::get(indices.getType().getShape(),
                                           rewriter.getI32Type());
  auto indices32 = rewriter.create<tosa::CastOp>(loc, indices32Ty, indices);

  auto scatterTy =
      RankedTensorType::get(scatterInOutShape, self.getType().getElementType());
  auto scatter =
      rewriter.create<tosa::ScatterOp>(loc, scatterTy, in, indices32, input);

  auto reshaped =
      mlir::tosa::reshapeTo(loc, rewriter, scatter, outType.getShape());

  rewriter.replaceOp(op, reshaped);
  return success();
}

// This defines a template to simplify legalization of certain ops.
template <typename AtenOpT>
class SimplifyAtenOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <>
LogicalResult SimplifyAtenOp<AtenConvolutionOp>::matchAndRewrite(
    AtenConvolutionOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // TOSA doesn't supports 1D convolutions.
  // We model them through a combination of AtenViewOp and 2D Convolution.
  // A Conv1D is replaced by:
  // %view = AtenViewOp (%input) : (3D type) -> (4D Type)
  // %conv2d = AtenConvolution (%view) : (4D type) -> (4D type)
  // %view2 = AtenViewOp (%conv2d) : (4D type) -> (3D type)

  auto inputTy = cast<RankedTensorType>(adaptor.getInput().getType());
  auto weightTy = cast<RankedTensorType>(adaptor.getWeight().getType());
  auto outputTy =
      cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

  auto ty = dyn_cast_or_null<BaseTensorType>(op.getType());
  if (!ty || !ty.hasSizes())
    return rewriter.notifyMatchFailure(
        op, "unimplemented: input must have known sizes");

  if (!inputTy || !weightTy || !outputTy)
    return rewriter.notifyMatchFailure(
        op, "Input, weight and output to Convolution must be ranked tensors");

  if (!weightTy.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: TOSA only supports static weight");

  if (inputTy.getRank() != 3)
    return rewriter.notifyMatchFailure(
        op, "Unimplemented: only simplify 1D convolution");

  auto loc = op->getLoc();

  auto getListConstructElementsPlusValue =
      [&](Value listConstruct, int64_t addedValue) -> std::optional<Value> {
    SmallVector<Value> values;
    if (!getListConstructElements(listConstruct, values)) {
      return std::nullopt;
    }

    Type ty = listConstruct.getType();
    values.push_back(
        rewriter.create<Torch::ConstantIntOp>(op->getLoc(), addedValue));
    return rewriter.create<PrimListConstructOp>(op->getLoc(), ty, values);
  };

  auto stride = getListConstructElementsPlusValue(op.getStride(), 1);
  if (!stride.has_value())
    return rewriter.notifyMatchFailure(op, "non-const stride list unsupported");

  auto dilation = getListConstructElementsPlusValue(op.getDilation(), 1);
  if (!dilation.has_value())
    return rewriter.notifyMatchFailure(op,
                                       "non-const dilation list unsupported");

  auto paddingValue = getListConstructElementsPlusValue(op.getPadding(), 0);
  if (!paddingValue.has_value())
    return rewriter.notifyMatchFailure(op,
                                       "non-const padding list unsupported");

  auto outputPaddingValue =
      getListConstructElementsPlusValue(op.getOutputPadding(), 0);
  if (!outputPaddingValue.has_value()) {
    return rewriter.notifyMatchFailure(
        op, "non-const output padding list unsupported");
  }

  auto addDimOneToSizes = [&](BaseTensorType ty) {
    SmallVector<int64_t> newSizes(ty.getSizes());
    newSizes.push_back(1);
    return newSizes;
  };

  auto input = op.getInput();
  auto weight = op.getWeight();

  auto newSizes = addDimOneToSizes(cast<BaseTensorType>(input.getType()));
  Value view1dTo2d = reshapeTo(loc, rewriter, input, newSizes);

  auto newWeightSizes =
      addDimOneToSizes(cast<BaseTensorType>(weight.getType()));
  weight = reshapeTo(loc, rewriter, weight, newWeightSizes);

  auto convSizes = addDimOneToSizes(cast<BaseTensorType>(ty));
  auto convTy = ty.getWithSizesAndDtype(convSizes, ty.getOptionalDtype());
  auto conv2dOp = rewriter.create<AtenConvolutionOp>(
      loc, convTy, view1dTo2d, weight, op.getBias(), *stride, *paddingValue,
      *dilation, op.getTransposed(), *outputPaddingValue, op.getGroups());

  Value view2dTo1d = reshapeTo(loc, rewriter, conv2dOp, ty.getSizes());
  rewriter.replaceOp(op, view2dTo1d);
  return success();
}

// The goal of this pattern is to handle the case where the indices for all
// dimensions except one are None.
class ConvertAtenIndexTensorOpNone
    : public OpConversionPattern<AtenIndexTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenIndexTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // To do so, we rewrite index.Tensor like that :
    // - To match tosa format of NxKxC, with K the dimension to extract from:
    //   - Transpose the dim to extract into position 'K'
    //   - flatten the other dimensions
    //   - Reshape to insert a 1x dimension as the N - The format should be
    //   1xKxC with C the flattened dimensions
    // - Insert a tosa.gather
    // - Bring back to the original format:
    //   - Reshape
    //   - Transpose
    auto loc = op->getLoc();
    auto outTy = dyn_cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));
    if (!outTy || !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op.getLoc(),
          "unimplemented: Only static shapes are currently supported");

    SmallVector<Value> torchIndices;
    if (!getListConstructElements(op.getIndices(), torchIndices))
      return rewriter.notifyMatchFailure(
          op.getLoc(),
          "unimplemented: the tensor list is not from list construct");

    auto indicesList =
        getTypeConvertedValues(rewriter, loc, typeConverter, torchIndices);

    // Check that all indices are none but one.
    int64_t indexDim = -1;
    for (size_t i = 0; i < indicesList.size(); ++i) {
      if (!indicesList[i])
        continue;
      if (indexDim != -1) {
        return rewriter.notifyMatchFailure(
            op.getLoc(), "unimplemented: only one dimension must be set in "
                         "indices for this pattern to work");
      }
      indexDim = i;
    }
    if (indexDim == -1) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "unimplemented: all indices are none");
    }

    auto indices =
        dyn_cast<TypedValue<RankedTensorType>>(indicesList[indexDim]);
    if (!indices) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "unimplemented: index must be ranked tensor");
    }

    if (indices.getType().getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "unimplemented: index must be 1d tensor");
    }

    auto input = adaptor.getSelf();
    auto inputTy = dyn_cast<RankedTensorType>(input.getType());
    if (!inputTy || !inputTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op.getLoc(), "unimplemented: input must have static shapes");
    auto inputElemTy = inputTy.getElementType();

    // Transpose indexDim into dimension 0
    SmallVector<int32_t> transposePerm;
    for (int64_t i = 0; i < inputTy.getRank(); ++i)
      transposePerm.push_back(i);
    transposePerm[0] = indexDim;
    transposePerm[indexDim] = 0;

    auto transposedInput =
        tosa::transposeBy(loc, rewriter, input, transposePerm);

    // Flatten matrix [k, ...] -> [1, k, c]
    auto transposedShape = transposedInput.getType().getShape();
    int64_t k = transposedShape[0];
    int64_t c =
        std::accumulate(transposedShape.begin() + 1, transposedShape.end(), 1,
                        [&](int64_t a, int64_t b) { return a * b; });

    SmallVector<int64_t> reshapedFormat = {1, k, c};
    // Reshapes the input to 1xKx(flattened_dims)
    auto reshapedInput =
        tosa::reshapeTo(loc, rewriter, transposedInput, reshapedFormat);

    auto w = indices.getType().getDimSize(0);
    auto reshapedIndices = tosa::reshapeTo(loc, rewriter, indices, {1, w});

    // And cast indices to i32
    TensorType promotedType = reshapedIndices.getType().cloneWith(
        reshapedIndices.getType().getShape(), rewriter.getI32Type());
    auto castedIndices = rewriter.create<tosa::CastOp>(
        op->getLoc(), promotedType, reshapedIndices);

    SmallVector<int64_t> gatherShape = {1, w, c};
    auto gatherOp = rewriter.create<tosa::GatherOp>(
        op->getLoc(), RankedTensorType::get(gatherShape, inputElemTy),
        reshapedInput, castedIndices);

    // Unflatten [1, w, c] -> [w, ...]
    SmallVector<int64_t> unflattenedShape{transposedShape};
    unflattenedShape[0] = w;
    auto unflattened =
        tosa::reshapeTo(loc, rewriter, gatherOp, unflattenedShape);

    SmallVector<int32_t> inversePermutation(transposePerm.size(), 0);
    for (size_t i = 0; i < transposePerm.size(); ++i)
      inversePermutation[transposePerm[i]] = i;

    // Transpose 'w' back in the original position of 'k'
    auto unTranspose =
        tosa::transposeBy(loc, rewriter, unflattened, inversePermutation);

    rewriter.replaceOp(op, unTranspose);
    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenIndexTensorHackedTwinOp>::matchAndRewrite(
    AtenIndexTensorHackedTwinOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // t        = tf.constant([[1, 2, 3, 4, 5],[6,7,8,9,10],
  //                         [11,12,13,14,15],[16,17,18,19,20]]) # 4*5
  // i        = tf.constant([[1,2,3], [3,2,1]]) # 2*3
  // i_expand = tf.expand_dims(i,axis=2) # 2*3*1
  // IndexTensorOutput = tf.gather_nd(t,tf.i_expand)
  //                   = torch.ops.aten.index(t, (i, )) = t[i] # 2*3*5
  // [[[ 6,  7,  8,  9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
  //  [[16, 17, 18, 19, 20], [11, 12, 13, 14, 15], [ 6,  7,  8,  9, 10]]]
  auto input = adaptor.getSelf();
  auto inputTensorType =
      dyn_cast<RankedTensorType>(adaptor.getSelf().getType());
  // Check input is a tensor type.
  if (!inputTensorType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  // Deal with torch.prim.ListConstruct of non const value to get the index
  auto tensorList = op.getIndices();
  SmallVector<Value> tensorsTorchType;
  if (!getListConstructElements(tensorList, tensorsTorchType))
    return op.emitError(
        "unimplemented: the tensor list is not from list construct");
  auto indexTensors = getTypeConvertedValues(
      rewriter, op->getLoc(), getTypeConverter(), tensorsTorchType);

  if (llvm::any_of(indexTensors, [](Value v) {
        auto tensorTy = dyn_cast<RankedTensorType>(v.getType());
        return !tensorTy || !tensorTy.hasStaticShape();
      })) {
    return rewriter.notifyMatchFailure(op, "expected static shape");
  }

  auto outType = getTypeConverter()->convertType(op.getType());

  // Support for multiple indexes
  if (indexTensors.size() > 1) {
    // t[i, i]
    // = torch.ops.aten.index(t,(i,i))
    // = tensor([[ t[1,1], t[2,2], t[3,3]],
    //           [ t[3,3], t[2,2],  t[1,1]]])
    // = tensor([[ 7, 13, 19], [19, 13,  7]])
    // = tf.gather_nd(t,tf.ii_expand)
    // ii_expand
    // = tf.concat((i_expand,i_expand), dim=2)
    // = tf.constant([[[1,1],[2,2],[3,3]],
    //                [[3,3],[2,2],[1,1]]]) # 2*3*2
    SmallVector<Value> indicesTfConcatTensors;
    SmallVector<int64_t> indexesRank;
    SmallVector<SmallVector<int64_t>> indexesShape;

    // concat index tensor into to indices tensor for concat
    for (size_t i = 0; i < indexTensors.size(); i++) {
      auto index = indexTensors[i];

      auto indexType = dyn_cast<RankedTensorType>(index.getType());
      auto indexShape = indexType.getShape();
      indexesShape.push_back(makeShapeTorchCompatible(indexShape));
      indexesRank.push_back(indexType.getRank());

      // Make type of index tosa compatible, i64 to i32.
      if (indexType.getElementType() != rewriter.getIntegerType(32)) {
        index = rewriter.create<tosa::CastOp>(
            op->getLoc(),
            RankedTensorType::get(indexShape, rewriter.getIntegerType(32)),
            index);
      }

      // Expand last dim of index to tf indices [2,3] -> [2,3,1]
      SmallVector<int64_t> indiceShapeOneDim;
      for (auto shape : indexShape) {
        indiceShapeOneDim.push_back(shape);
      }
      indiceShapeOneDim.push_back(1);
      auto indicesTfOneDim = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
          rewriter, op->getLoc(),
          RankedTensorType::get(indiceShapeOneDim, rewriter.getIntegerType(32)),
          index, rewriter.getDenseI64ArrayAttr(indiceShapeOneDim));

      // create concat tensor for indicesTf
      indicesTfConcatTensors.push_back(indicesTfOneDim.getResult());
    }

    auto getRankExtendedShape =
        [](SmallVector<int64_t> inputShape,
           SmallVector<int64_t> maxRank1DimShape) -> SmallVector<int64_t> {
      SmallVector<int64_t> rankExtendedShape(maxRank1DimShape);
      auto inputRank = inputShape.size();
      auto maxRank = maxRank1DimShape.size();
      auto startIdx = maxRank - inputRank;
      for (size_t i = startIdx; i < maxRank; i++) {
        rankExtendedShape[i] = inputShape[i - startIdx];
      }
      return rankExtendedShape;
    };

    bool hasDiffShapedIndexes = false;
    for (auto indexShapeOneDim : indexesShape) {
      if (!llvm::equal(indexesShape[0], indexShapeOneDim)) {
        hasDiffShapedIndexes = true;
        break;
      }
    }

    if (hasDiffShapedIndexes) {
      int64_t maxRank = 1;
      for (auto idxRank : indexesRank) {
        if (idxRank > maxRank)
          maxRank = idxRank;
      }
      // Tensor shape of max rank, each dim being 1
      SmallVector<int64_t> maxRank1DimShape;
      for (int i = 0; i < maxRank; i++)
        maxRank1DimShape.push_back(1);
      // Tensor shape of max rank, each dim being the max dim.
      SmallVector<int64_t> maxRankMaxDimShape(maxRank1DimShape);

      auto updateMaxRankMaxDimShape =
          [&](SmallVector<int64_t> broadcastedShape) -> LogicalResult {
        for (size_t i = 0; i < maxRankMaxDimShape.size(); i++) {
          // check for malformed index tensors
          if (broadcastedShape[i] != 1 && maxRankMaxDimShape[i] != 1 &&
              maxRankMaxDimShape[i] != broadcastedShape[i]) {
            return failure();
          }
          if (broadcastedShape[i] > maxRankMaxDimShape[i])
            maxRankMaxDimShape[i] = broadcastedShape[i];
        }
        return success();
      };

      for (size_t i = 0; i < indexesRank.size(); i++) {
        // Reshape all index tensors to same maxRank
        auto idxRank = indexesRank[i];
        auto unreshapedIdxTensor = indicesTfConcatTensors[i];
        SmallVector<int64_t> broadcastedShape =
            getRankExtendedShape(indexesShape[i], maxRank1DimShape);

        if (idxRank < maxRank) {
          auto idxType =
              dyn_cast<RankedTensorType>(indicesTfConcatTensors[i].getType());
          // indicesTfConcatTensors has a trailing [1] dim for the final concat.
          auto broadcastedShapeTf(broadcastedShape);
          broadcastedShapeTf.push_back(1);
          auto reshapeOutputTy = RankedTensorType::get(
              broadcastedShapeTf, idxType.getElementType());
          // Update the tensor array with the max rank-extended form
          indicesTfConcatTensors[i] = rewriter.create<tosa::ReshapeOp>(
              op->getLoc(), reshapeOutputTy, unreshapedIdxTensor,
              rewriter.getDenseI64ArrayAttr(broadcastedShapeTf));
        }

        // Construct the max rank broadcasted form of all index tensors with
        // each index tensor.
        if (updateMaxRankMaxDimShape(broadcastedShape).failed()) {
          return rewriter.notifyMatchFailure(
              op, "Malformed index tensors that have mismatched dim shapes");
        }

        // Every index now has the same rank but not yet same shape until
        // tosa.tile below.
        indexesShape[i] = broadcastedShape;
        indexesRank[i] = maxRank;
      }

      auto getTileOpShape = [&](SmallVector<int64_t> indexShape,
                                SmallVector<int64_t> &tileOpShape) -> bool {
        bool needsTiling = false;
        for (size_t i = 0; i < indexShape.size(); i++) {
          if (1 == indexShape[i]) {
            tileOpShape.push_back(maxRankMaxDimShape[i]);
            needsTiling = true;
          } else {
            tileOpShape.push_back(1);
          }
        }
        return needsTiling;
      };

      // Use tosa.tile to broadcast in multiple dims so all index tensors have
      // the same shape. This materializes new tensors.
      for (size_t i = 0; i < indexesRank.size(); i++) {
        SmallVector<int64_t> tileOpShape;
        bool needsTiling = getTileOpShape(indexesShape[i], tileOpShape);

        if (needsTiling) {
          auto idxType =
              dyn_cast<RankedTensorType>(indicesTfConcatTensors[i].getType());
          // indicesTfConcatTensors has a trailing [1] dim for the final concat.
          auto maxRankMaxDimShapeTf(maxRankMaxDimShape);
          maxRankMaxDimShapeTf.push_back(1);
          auto tileOpShapeTf(tileOpShape);
          tileOpShapeTf.push_back(1);
          auto tileOutputTy = RankedTensorType::get(maxRankMaxDimShapeTf,
                                                    idxType.getElementType());
          auto reshapedIdxTensor = indicesTfConcatTensors[i];
          indicesTfConcatTensors[i] = rewriter.create<tosa::TileOp>(
              op->getLoc(), tileOutputTy, reshapedIdxTensor,
              rewriter.getDenseI64ArrayAttr(tileOpShapeTf));
        }

        // Every index tensor now has the same rank and shape
        indexesShape[i] = maxRankMaxDimShape;
      }
    }

    // concat each indices into indicesTf: shape [2,3,1],[2,3,1] -> [2,3,2]
    auto indicesShapeConcat = indexesShape[0];
    uint64_t lastDim = indexesRank[0];
    indicesShapeConcat.push_back(indicesTfConcatTensors.size());
    auto indicesTf = tosa::CreateOpAndInfer<tosa::ConcatOp>(
        rewriter, op->getLoc(),
        GetTypeFromTensorShape(indicesShapeConcat, rewriter.getIntegerType(32)),
        indicesTfConcatTensors, lastDim);

    if (!indicesTf) {
      return rewriter.notifyMatchFailure(
          op, "Convert TorchIndex To TfIndices fail.");
    }
    // do the tf gathernp algorithm with tf style indices as input.
    auto result = tosa::convertGatherNdOp(rewriter, op, outType, input,
                                          indicesTf.getResult());

    if (!result) {
      return rewriter.notifyMatchFailure(
          op, "Convert GatherNdOp fail for index tensor.");
    }
    rewriter.replaceOp(op, {result.value()});

    return success();
  }

  // Support for multiple index
  auto index = indexTensors[0];
  auto indexType = dyn_cast<RankedTensorType>(index.getType());
  auto indexShape = indexType.getShape();
  // index i64 to i32 for tosa compatible
  if (indexType.getElementType() != rewriter.getIntegerType(32)) {
    index = rewriter.create<tosa::CastOp>(
        op->getLoc(),
        RankedTensorType::get(indexShape, rewriter.getIntegerType(32)), index);
  }

  // Expand last dim of index to tf indices [2,3] -> [2,3,1]
  SmallVector<int64_t> indicesShape;
  for (auto shape : indexShape) {
    indicesShape.push_back(shape);
  }
  indicesShape.push_back(1);
  auto indicesTf = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
      rewriter, op->getLoc(),
      RankedTensorType::get(indicesShape, rewriter.getIntegerType(32)), index,
      rewriter.getDenseI64ArrayAttr(indicesShape));

  if (!indicesTf) {
    return rewriter.notifyMatchFailure(op,
                                       "Convert TorchIndex To TfIndices fail.");
  }
  // do the tf gathernp algorithm with tf style indices as input.
  auto result = tosa::convertGatherNdOp(rewriter, op, outType, input,
                                        indicesTf.getResult());

  if (!result) {
    return rewriter.notifyMatchFailure(
        op, "Convert GatherNdOp fail for index tensor.");
  }
  rewriter.replaceOp(op, {result.value()});

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenAbsOp>::matchAndRewrite(
    AtenAbsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");

  auto outType = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<tosa::AbsOp>(op, outType, adaptor.getSelf());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenWhereSelfOp>::matchAndRewrite(
    AtenWhereSelfOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");
  auto condType = dyn_cast<TensorType>(adaptor.getCondition().getType());
  if (!condType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types condition are currently supported");

  auto outType = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<tosa::SelectOp>(
      op, outType, adaptor.getCondition(), adaptor.getSelf(),
      adaptor.getOther());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenIscloseOp>::matchAndRewrite(
    AtenIscloseOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // check args
  double rtol, atol;
  bool equalNan;
  if (!matchPattern(op.getRtol(), m_TorchConstantFloat(&rtol)))
    return rewriter.notifyMatchFailure(op, "rtol must be a scalar constant");
  if (!matchPattern(op.getAtol(), m_TorchConstantFloat(&atol)))
    return rewriter.notifyMatchFailure(op, "atol must be a scalar constant");
  if (!matchPattern(op.getEqualNan(), m_TorchConstantBool(&equalNan)))
    return rewriter.notifyMatchFailure(
        op, "unimplemented: equal_nan is expected to be false");

  // check tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  auto otherType = dyn_cast<TensorType>(adaptor.getOther().getType());
  if (!selfType || !otherType)
    return rewriter.notifyMatchFailure(
        op, "Only tensor types input are currently supported");
  if (!selfType.hasStaticShape() || !otherType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");
  if (!isa<mlir::FloatType>(selfType.getElementType()) ||
      !isa<mlir::FloatType>(otherType.getElementType())) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: only FP element type is supported");
  }

  auto rhsSubOp = rewriter.create<tosa::SubOp>(
      op->getLoc(), selfType, adaptor.getSelf(), adaptor.getOther());
  auto rhsAbsOp =
      rewriter.create<tosa::AbsOp>(op->getLoc(), selfType, rhsSubOp);

  auto lhsAbsOp =
      rewriter.create<tosa::AbsOp>(op->getLoc(), otherType, adaptor.getOther());
  auto rtolConstOp =
      tosa::getTosaConstTensorSingleF32(rewriter, op, static_cast<float>(rtol));
  auto mulOp = rewriter.create<tosa::MulOp>(op->getLoc(), otherType,
                                            rtolConstOp, lhsAbsOp, /*shift=*/0);
  auto atolConstOp =
      tosa::getTosaConstTensorSingleF32(rewriter, op, static_cast<float>(atol));
  auto addOp =
      rewriter.create<tosa::AddOp>(op->getLoc(), otherType, atolConstOp, mulOp);

  auto outType = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<tosa::GreaterEqualOp>(op, outType, addOp,
                                                    rhsAbsOp);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenClampOp>::matchAndRewrite(
    AtenClampOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType)
    return rewriter.notifyMatchFailure(
        op, "only tensor types input are currently supported");

  IntegerAttr min_int =
      rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::min());
  IntegerAttr max_int =
      rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
  FloatAttr min_fp =
      rewriter.getF32FloatAttr(std::numeric_limits<float>::lowest());
  FloatAttr max_fp =
      rewriter.getF32FloatAttr(std::numeric_limits<float>::max());

  auto getValAttr = [&](Value operand, IntegerAttr &intAttr,
                        FloatAttr &fpAttr) -> LogicalResult {
    double valFloat;
    int64_t valInt;
    if (matchPattern(operand, m_TorchConstantFloat(&valFloat))) {
      intAttr = rewriter.getI64IntegerAttr(static_cast<int64_t>(valFloat));
      fpAttr = rewriter.getF32FloatAttr(static_cast<float>(valFloat));
    } else if (matchPattern(operand, m_TorchConstantInt(&valInt))) {
      intAttr = rewriter.getI64IntegerAttr(valInt);
      fpAttr = rewriter.getF32FloatAttr(static_cast<float>(valInt));
    } else {
      return failure();
    }
    return success();
  };

  LogicalResult minAttrResult = getValAttr(op.getMin(), min_int, min_fp);
  LogicalResult maxAttrResult = getValAttr(op.getMax(), max_int, max_fp);
  if (failed(minAttrResult) && failed(maxAttrResult)) {
    return rewriter.notifyMatchFailure(
        op, "either `min` or `max` should be a torch constant");
  }
  if (failed(minAttrResult) &&
      succeeded(checkNotNone(rewriter, op, op.getMin()))) {
    return rewriter.notifyMatchFailure(op,
                                       "min attr should be a torch constant");
  }
  if (failed(maxAttrResult) &&
      succeeded(checkNotNone(rewriter, op, op.getMax()))) {
    return rewriter.notifyMatchFailure(op,
                                       "max attr should be a torch constant");
  }

  auto outType = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<tosa::ClampOp>(op, outType, adaptor.getSelf(),
                                             min_int, max_int, min_fp, max_fp);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenArangeStartStepOp>::matchAndRewrite(
    AtenArangeStartStepOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  const TypeConverter *typeConverter = this->getTypeConverter();
  RankedTensorType resultType = cast<RankedTensorType>(
      typeConverter->convertType(op->getResult(0).getType()));

  // At this point all tensors should have value semantics, and hence the
  // `layout` check can be ignored.

  // TODO: Add support for pin_memory features.
  // The pin_memory should be either `False` or `none`.
  bool pinMemory;
  if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
      (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
       pinMemory)) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: pin_memory must be either None or false");
  }

  // Stores a range value (a start, end, or step value) and whether or not it
  // was initiated with a constant integer, an constant float or neither.
  class ConstRangeValue {
  public:
    explicit ConstRangeValue(double v)
        : vDouble(v), fromDouble(true), vInt(static_cast<int64_t>(v)),
          fromInt(false) {}

    explicit ConstRangeValue(int64_t v)
        : vDouble(static_cast<double>(v)), fromDouble(false), vInt(v),
          fromInt(true) {}

    // Constructor for the case where there is no constant value to use.
    ConstRangeValue()
        : vDouble(0), fromDouble(false), vInt(0), fromInt(false) {}

    static ConstRangeValue fromValue(Value v) {
      int64_t intVal{0};
      double floatVal{0.0};
      if (matchPattern(v, m_TorchConstantFloat(&floatVal))) {
        return ConstRangeValue(floatVal);
      } else if (matchPattern(v, m_TorchConstantInt(&intVal))) {
        return ConstRangeValue(intVal);
      }
      return ConstRangeValue();
    }

    bool hasConstInt() const { return fromInt; }
    bool hasConstDouble() const { return fromDouble; }
    bool hasConst() const { return fromInt || fromDouble; }
    double getDouble() const { return vDouble; }
    int64_t getInt() const { return vInt; }

  private:
    double vDouble;
    bool fromDouble;
    int64_t vInt;
    bool fromInt;
  };

  auto start = ConstRangeValue::fromValue(op.getStart());
  if (!start.hasConst()) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: case where `start` is not a constant int or float");
  }

  auto end = ConstRangeValue::fromValue(op.getEnd());
  if (!end.hasConst()) {
    return rewriter.notifyMatchFailure(
        op,
        "unimplemented: case where value `end` is not a constant int or float");
  }

  auto step = ConstRangeValue::fromValue(op.getStep());
  if (!step.hasConst()) {
    return rewriter.notifyMatchFailure(op,
                                       "unimplemented: case where value `step` "
                                       "is not a constant int or float");
  }

  auto getRange = [](auto start, auto end, auto step) {
    // Initialize a small vector of the same type as start:
    using T = decltype(start);
    SmallVector<T> values;

    uint64_t counter{0};
    if (start == end) {
      return values;
    }
    assert(step != T(0));
    values.reserve(
        1 + static_cast<size_t>(std::abs((end - start) / std::abs(step))));
    if (step > 0) {
      while (start + T(counter) * step < end) {
        values.push_back(start + counter * step);
        counter++;
      }
    } else {
      while (start + T(counter) * step > end) {
        values.push_back(start + counter * step);
        counter++;
      }
    }
    return values;
  };

  const auto isIntType =
      dyn_cast_or_null<mlir::IntegerType>(resultType.getElementType());

  const auto isDoubleType =
      dyn_cast_or_null<mlir::FloatType>(resultType.getElementType());

  auto maybeResult = [&]() -> std::optional<Value> {
    // Integer output type, and start / end / range are all integers.
    if (isIntType && start.hasConstInt() && end.hasConstInt() &&
        step.hasConstInt()) {
      auto values = getRange(start.getInt(), end.getInt(), step.getInt());
      return tosa::getConstTensor<int64_t>(rewriter, op, values, values.size());
    }

    // Get a double range.
    auto values =
        getRange(start.getDouble(), end.getDouble(), step.getDouble());
    if (isIntType) {
      SmallVector<int64_t> values_i64;
      values_i64.reserve(values.size());
      for (auto v : values) {
        values_i64.push_back(static_cast<int64_t>(v));
      }
      return tosa::getConstTensor<int64_t>(rewriter, op, values_i64,
                                           values.size());
    }

    if (!isDoubleType) {
      return {};
    }

    SmallVector<float> values_f32;
    values_f32.reserve(values.size());
    for (auto v : values) {
      values_f32.push_back(static_cast<float>(v));
    }
    auto vs = tosa::getConstTensor<float>(rewriter, op, values_f32,
                                          values_f32.size());
    return vs;
  }();

  if (!maybeResult.has_value()) {
    return rewriter.notifyMatchFailure(
        op, "failed to generate constant tensor for arange");
  }
  auto result = maybeResult.value();

  rewriter.replaceOpWithNewOp<tosa::CastOp>(op, resultType, result);
  return success();
}

template <>
LogicalResult ConvertAtenOp<PrimNumToTensorScalarOp>::matchAndRewrite(
    PrimNumToTensorScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  const TypeConverter *typeConverter = this->getTypeConverter();
  RankedTensorType resultType = cast<RankedTensorType>(
      typeConverter->convertType(op->getResult(0).getType()));

  // Only supports integer operand type, because for the floating point operand
  // type result tensor has to be of type `f64` which is not supported in the
  // tosa.
  double doubleValue;
  auto isDouble = matchPattern(op.getA(), m_TorchConstantFloat(&doubleValue));
  int64_t intValue;
  auto isInt = matchPattern(op.getA(), m_TorchConstantInt(&intValue));
  if (!isDouble && !isInt)
    return rewriter.notifyMatchFailure(op,
                                       "Unable to extract the scalar constant");

  auto outElemTy = resultType.getElementType();
  if (isa<mlir::IntegerType>(outElemTy)) {
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(
        op, resultType, DenseElementsAttr::get(resultType, {intValue}));
  } else if (outElemTy.isF64()) {
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(
        op, resultType, DenseElementsAttr::get(resultType, {doubleValue}));
  }

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenCopyOp>::matchAndRewrite(
    AtenCopyOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  auto srcType = dyn_cast<TensorType>(adaptor.getSrc().getType());
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  if (!srcType || !srcType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  // The non_blocking should be a constant `False`.
  bool nonBlocking;
  if (!matchPattern(op.getNonBlocking(), m_TorchConstantBool(&nonBlocking))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking must be a constant");
  } else if (nonBlocking) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking is expected to be false");
  }

  SmallVector<int64_t> selfShape(makeShapeTorchCompatible(selfType.getShape()));
  SmallVector<int64_t> srcShape(makeShapeTorchCompatible(srcType.getShape()));

  if (llvm::equal(selfShape, srcShape) || selfShape.size() == 0) {
    // If we reach here, then it means the given case is handled by implicit
    // broadcasting done by tosa.
    Value result;
    if (failed(tosa::tosaCastTensorToType(
            rewriter, op, adaptor.getSrc(),
            getTypeConverter()->convertType(op.getType()), result)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: cast to result type not supported");
    rewriter.replaceOp(op, result);
    return success();
  }
  return rewriter.notifyMatchFailure(
      op, "unimplemented: valsem.aten.copy op not supported for this case.");
}

//  Legalizes the torch.aten.to.dtype op
template <>
LogicalResult ConvertAtenOp<AtenToDtypeOp>::matchAndRewrite(
    AtenToDtypeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Not a tensor type.
  auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
  if (!selfType || !selfType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "Only tensor types with static shape are supported");

  // The non_blocking arg should be a constant `False`.
  bool nonBlocking;
  if (!matchPattern(op.getNonBlocking(), m_TorchConstantBool(&nonBlocking))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking arg must be a constant");
  } else if (nonBlocking) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: non_blocking arg is expected to be false");
  }

  // The copy arg should be a constant `False`.
  bool copy;
  if (!matchPattern(op.getCopy(), m_TorchConstantBool(&copy))) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: copy arg must be a constant");
  } else if (copy) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: copy arg is expected to be false");
  }

  // Only `none`, `contiguous` and `preserve` memory_format is supported.
  if (!isa<Torch::NoneType>(op.getMemoryFormat().getType())) {
    int64_t memoryFormat;
    if (!matchPattern(op.getMemoryFormat(), m_TorchConstantInt(&memoryFormat)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the memory format should be specified in "
              "an integer constant");
    if (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
        memoryFormat != torch_upstream::MemoryFormat::Preserve)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only none, contiguous and preserve "
              "memory_format is supported");
  }

  auto resultTy = cast<RankedTensorType>(
      getTypeConverter()->convertType(op.getResult().getType()));

  Value result;
  if (failed(tosa::tosaCastTensorToType(rewriter, op, adaptor.getSelf(),
                                        resultTy, result)))
    return rewriter.notifyMatchFailure(op, "conversion to result type failed");

  rewriter.replaceOp(op, result);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenRemainderScalarOp>::matchAndRewrite(
    AtenRemainderScalarOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  Value self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());

  if (!selfTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor types supported in TOSA Remainder");

  auto outType =
      cast<TensorType>(getTypeConverter()->convertType(op.getType()));

  Type outElemTy = outType.getElementType();
  if (!outElemTy.isIntOrFloat())
    return rewriter.notifyMatchFailure(
        op, "Only floating-point or integer datatype legalization supported");

  Value otherTensor;
  Value other = op.getOther();
  if (failed(torchScalarToTosaTensor(rewriter, op, other, otherTensor,
                                     outElemTy, {})))
    return rewriter.notifyMatchFailure(
        op, "Currently only scalar constants are supported for "
            "conversion in TOSA Remainder operation");

  if (selfTy.getElementType() != outElemTy)
    self = rewriter.create<tosa::CastOp>(op.getLoc(), outType, self);

  auto divTensor = self;
  if (isa<mlir::FloatType>(outElemTy)) {
    auto otherTensorReciprocal = rewriter.create<tosa::ReciprocalOp>(
        op.getLoc(), otherTensor.getType(), otherTensor);
    divTensor = rewriter.create<tosa::MulOp>(
        op.getLoc(), outType, self, otherTensorReciprocal, /*shift=*/0);
    divTensor = rewriter.create<tosa::FloorOp>(op.getLoc(), outType, divTensor);
  } else {
    divTensor = rewriter.create<tosa::IntDivOp>(op.getLoc(), outType, self,
                                                otherTensor);
  }

  auto mulTensor =
      rewriter.create<tosa::MulOp>(op.getLoc(), outType, otherTensor, divTensor,
                                   /*shift=*/0);
  rewriter.replaceOpWithNewOp<tosa::SubOp>(op, outType, self, mulTensor);

  return success();
}

template <typename AtenOpT, typename TosaOpT>
class ConvertAtenPoolingBaseOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  // Different pooling variants need to process inputs differently, e.g.
  // adaptive pooling generates the kernel size rather than receive it. This
  // function also transposes inputs.
  virtual LogicalResult processInputs(AtenOpT op, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter,
                                      Value &input, DenseI64ArrayAttr &kernel,
                                      DenseI64ArrayAttr &stride,
                                      DenseI64ArrayAttr &pad,
                                      Type &outputTy) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented pooling input parsing function");
  }

  static int64_t getOutputDim(int64_t inputDim, int64_t kernelDim,
                              int64_t stride, int64_t padBefore,
                              int64_t padAfter, int64_t dilation,
                              bool ceilMode = false) {
    if (inputDim == kUnknownSize) {
      return kUnknownSize;
    } else {
      int64_t dimSize =
          inputDim + padBefore + padAfter - dilation * (kernelDim - 1) - 1;
      if (ceilMode && (dimSize % stride != 0))
        return dimSize / stride + 2;
      return dimSize / stride + 1;
    }
  }

  // Apply the transposeDims vector on input to generate a transposed form.
  Value transposeTensor(AtenOpT op, ConversionPatternRewriter &rewriter,
                        Value input, ArrayRef<int32_t> transposeDims) const {
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto inputElemTy = inputTy.getElementType();
    auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
    auto inputRank = inputTy.getRank();

    std::optional<Value> transposeDimsConst = tosa::getConstTensor<int32_t>(
        rewriter, op,
        /*vec=*/transposeDims,
        /*shape=*/{static_cast<int32_t>(inputRank)});

    SmallVector<int64_t> transposedInputShape;
    for (auto &dim : transposeDims)
      transposedInputShape.push_back(inputShape[dim]);
    auto transposedInputType = RankedTensorType::get(
        makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
    return rewriter
        .create<tosa::TransposeOp>(op->getLoc(), transposedInputType, input,
                                   transposeDimsConst.value())
        .getResult();
  }

  Value transposePoolingInputToHwc(AtenOpT op,
                                   ConversionPatternRewriter &rewriter,
                                   Value input) const {
    auto inputRank = cast<RankedTensorType>(input.getType()).getRank();

    SmallVector<int32_t> nchwToNhwc4DTransposeDims({0, 2, 3, 1});
    SmallVector<int32_t> chwToHwc3DTransposeDims({1, 2, 0});

    return transposeTensor(op, rewriter, input,
                           inputRank == 3 ? chwToHwc3DTransposeDims
                                          : nchwToNhwc4DTransposeDims);
  }

  Value transposePoolingOutputToChw(AtenOpT op,
                                    ConversionPatternRewriter &rewriter,
                                    Value input) const {
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto inputRank = inputTy.getRank();

    SmallVector<int32_t> nhwcToNchw4DTransposeDims({0, 3, 1, 2});
    SmallVector<int32_t> hwcToChw3DTransposeDims({2, 0, 1});

    return transposeTensor(op, rewriter, input,
                           inputRank == 3 ? hwcToChw3DTransposeDims
                                          : nhwcToNchw4DTransposeDims);
  }

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input;
    DenseI64ArrayAttr kernel, stride, pad;
    Type outputTy;

    // Attempts to read input and kernel parameters, or synthesize them in the
    // case of adaptive pooling. Also performs input CHW->HWC transpose.
    if (failed(processInputs(op, adaptor, rewriter, input, kernel, stride, pad,
                             outputTy)))
      return rewriter.notifyMatchFailure(
          op, "Failed to process inputs for pooling");

    Value pooledOutput;
    static_assert(std::is_same<TosaOpT, tosa::MaxPool2dOp>::value ||
                      std::is_same<TosaOpT, tosa::AvgPool2dOp>::value,
                  "Expected either tosa::MaxPool2dOp or tosa::AvgPool2dOp");
    if constexpr (std::is_same<TosaOpT, tosa::MaxPool2dOp>::value) {
      pooledOutput = rewriter
                         .create<TosaOpT>(op->getLoc(), outputTy, input, kernel,
                                          stride, pad)
                         .getResult();
    } else if constexpr (std::is_same<TosaOpT, tosa::AvgPool2dOp>::value) {
      TypeAttr accType;
      if (failed(tosa::getAvgPool2dAccType(rewriter, input, accType)))
        return rewriter.notifyMatchFailure(
            op, "Failed to get accumulator type for pooling");
      pooledOutput = rewriter
                         .create<TosaOpT>(op->getLoc(), outputTy, input, kernel,
                                          stride, pad, accType)
                         .getResult();
    }

    auto transposedOutput =
        ConvertAtenPoolingBaseOp<AtenOpT, TosaOpT>::transposePoolingOutputToChw(
            op, rewriter, pooledOutput);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        transposedOutput);

    return success();
  }
};

template <typename AtenOpT, typename TosaOpT>
class ConvertAtenAdaptivePoolingOp
    : public ConvertAtenPoolingBaseOp<AtenOpT, TosaOpT> {
public:
  using ConvertAtenPoolingBaseOp<AtenOpT, TosaOpT>::ConvertAtenPoolingBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult processInputs(AtenOpT op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              DenseI64ArrayAttr &kernel,
                              DenseI64ArrayAttr &stride, DenseI64ArrayAttr &pad,
                              Type &outputTy) const override {
    auto inputXchw = adaptor.getSelf();
    auto inputTy = cast<RankedTensorType>(inputXchw.getType());
    if (!inputTy)
      return rewriter.notifyMatchFailure(
          op, "Adaptive avgpool requires ranked tensor input");

    auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
    auto inputRank = inputTy.getRank();
    auto inputElemTy = inputTy.getElementType();

    // Rank sanity check.
    if (inputTy.getRank() != 4 && inputRank != 3)
      return rewriter.notifyMatchFailure(
          op, "NCHW->NHWC transpose requires 3D or 4D tensor");

    int64_t inputHDim = inputShape[inputRank - 2];
    int64_t inputWDim = inputShape[inputRank - 1];

    SmallVector<int64_t> outputSize;
    if (!matchPattern(op.getOutputSize(),
                      m_TorchListOfConstantInts(outputSize)))
      return rewriter.notifyMatchFailure(
          op, "Non-const output_size for adaptive pooling unsupported.");

    SmallVector<int64_t> kernelDims;
    int64_t outputHDim, outputWDim;
    if (outputSize.size() == 1) {
      outputHDim = outputWDim = outputSize[0];
    } else {
      if (outputSize.size() != 2)
        return rewriter.notifyMatchFailure(
            op, "Adaptive avgpool output_size not 1 or 2 elements.");

      // Assumes 'None' (e.g. output_size=(None, 5) ) is expressed as <=0.
      outputHDim =
          (outputSize[0] <= 0) ? inputShape[inputRank - 2] : outputSize[0];
      outputWDim =
          (outputSize[1] <= 0) ? inputShape[inputRank - 1] : outputSize[1];
    }

    // In adaptive pooling,
    // stride = inputDim // outputDim
    // kernel = inputDim - (outputDim-1)* stride
    // pad = 0, dilation = 1

    int64_t strideH = inputShape[inputRank - 2] / outputHDim;
    int64_t strideW = inputShape[inputRank - 1] / outputWDim;

    kernelDims.push_back(inputHDim - (outputHDim - 1) * strideH);
    kernelDims.push_back(inputWDim - (outputWDim - 1) * strideW);

    SmallVector<int64_t> outputShape;
    if (inputRank > 3)
      outputShape.push_back(inputShape[0]);
    outputShape.push_back(outputHDim);
    outputShape.push_back(outputWDim);
    outputShape.push_back(inputShape[inputRank - 3]);

    // Transpose to xHWC
    input =
        ConvertAtenPoolingBaseOp<AtenOpT, TosaOpT>::transposePoolingInputToHwc(
            op, rewriter, inputXchw);
    kernel = rewriter.getDenseI64ArrayAttr(kernelDims);
    stride = rewriter.getDenseI64ArrayAttr({strideH, strideW});
    // Adaptive pooling does unit dilation and zero pad.
    pad = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
    outputTy = RankedTensorType::get(makeShapeLLVMCompatible(outputShape),
                                     inputElemTy);

    return success();
  }
};

template <typename AtenOpT, typename tosaOp>
static Type getOutputTypeForNonAdaptivePoolingOp(
    RankedTensorType inputTy, SmallVectorImpl<int64_t> &kernelSize,
    SmallVectorImpl<int64_t> &strideArray, SmallVectorImpl<int64_t> &padArray,
    SmallVectorImpl<int64_t> &dilationArray, bool ceilMode = false) {
  auto inputShape = makeShapeTorchCompatible(inputTy.getShape());
  auto inputRank = inputTy.getRank();
  auto inputElemTy = inputTy.getElementType();

  int64_t outputHDim = ConvertAtenPoolingBaseOp<AtenOpT, tosaOp>::getOutputDim(
      inputShape[inputRank - 2], kernelSize[0], strideArray[0], padArray[0],
      padArray[0], dilationArray[0], ceilMode);
  int64_t outputWDim = ConvertAtenPoolingBaseOp<AtenOpT, tosaOp>::getOutputDim(
      inputShape[inputRank - 1], kernelSize[1], strideArray[1], padArray[1],
      padArray[1], dilationArray[1], ceilMode);
  padArray[0] = (outputHDim - 1) * strideArray[0] +
                dilationArray[0] * kernelSize[0] - dilationArray[0] + 1 -
                padArray[0] * 2 - inputShape[inputRank - 2];
  padArray[1] = (outputWDim - 1) * strideArray[1] +
                dilationArray[0] * kernelSize[1] - dilationArray[0] + 1 -
                padArray[1] * 2 - inputShape[inputRank - 1];
  SmallVector<int64_t> outputShape;
  if (inputRank > 3)
    outputShape.push_back(inputShape[0]);
  outputShape.push_back(outputHDim);
  outputShape.push_back(outputWDim);
  outputShape.push_back(inputShape[inputRank - 3]);
  return RankedTensorType::get(makeShapeLLVMCompatible(outputShape),
                               inputElemTy);
}

// Checks the validity of pooling parameters and stores them in the respective
// vector. Also, gets the output type for the pooling op.
template <typename AtenOpT, typename tosaOp>
static LogicalResult getOutputTypeAndPoolingParameters(
    AtenOpT op, ConversionPatternRewriter &rewriter, Value inputXchw,
    SmallVectorImpl<int64_t> &dilationArray, Type &outputTy,
    DenseI64ArrayAttr &kernel, DenseI64ArrayAttr &stride,
    DenseI64ArrayAttr &pad) {

  RankedTensorType inputTy = cast<RankedTensorType>(inputXchw.getType());
  if (!inputTy)
    return rewriter.notifyMatchFailure(
        op, "Pooling op requires ranked tensor input");

  auto inputRank = inputTy.getRank();
  // Rank sanity check.
  if (inputTy.getRank() != 4 && inputRank != 3)
    return rewriter.notifyMatchFailure(
        op, "NCHW->NHWC transpose requires 3D or 4D tensor");

  SmallVector<int64_t, 2> kernelSizeInts, strideInts, paddingInts;
  if (!matchPattern(op.getKernelSize(),
                    m_TorchListOfConstantInts(kernelSizeInts)))
    return rewriter.notifyMatchFailure(
        op, "Non-const kernel_size for pooling op unsupported");

  if (!matchPattern(op.getStride(), m_TorchListOfConstantInts(strideInts)))
    return rewriter.notifyMatchFailure(
        op, "Non-const stride for pooling op unsupported");
  // If `stride` is not specified by the user, it is assigned the value of empty
  // list during import. For such a case, the stride value is the kernel size.
  // See:
  // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
  if (strideInts.empty())
    strideInts.assign(kernelSizeInts);

  if (!matchPattern(op.getPadding(), m_TorchListOfConstantInts(paddingInts)))
    return rewriter.notifyMatchFailure(
        op, "Non-const padding factor for pooling op unsupported");

  SmallVector<int64_t, 4> padArr = {paddingInts[0], paddingInts[0],
                                    paddingInts[1], paddingInts[1]};
  kernel = rewriter.getDenseI64ArrayAttr(kernelSizeInts);
  stride = rewriter.getDenseI64ArrayAttr(strideInts);

  bool ceilMode;
  if (!matchPattern(op.getCeilMode(), m_TorchConstantBool(&ceilMode)))
    return rewriter.notifyMatchFailure(
        op, "only support constant bool ceil_mode for pooling op");

  outputTy = getOutputTypeForNonAdaptivePoolingOp<AtenOpT, tosaOp>(
      inputTy, kernelSizeInts, strideInts, paddingInts, dilationArray,
      ceilMode);
  padArr[1] = padArr[1] + paddingInts[0];
  padArr[3] = padArr[3] + paddingInts[1];
  pad = rewriter.getDenseI64ArrayAttr(
      {padArr[0], padArr[1], padArr[2], padArr[3]});
  return success();
}

class ConvertAtenMaxPool2dOp
    : public ConvertAtenPoolingBaseOp<AtenMaxPool2dOp, tosa::MaxPool2dOp> {
public:
  using ConvertAtenPoolingBaseOp<AtenMaxPool2dOp,
                                 tosa::MaxPool2dOp>::ConvertAtenPoolingBaseOp;
  LogicalResult processInputs(AtenMaxPool2dOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              DenseI64ArrayAttr &kernel,
                              DenseI64ArrayAttr &stride, DenseI64ArrayAttr &pad,
                              Type &outputTy) const override {
    SmallVector<int64_t, 2> dilationArray;
    if (!matchPattern(op.getDilation(),
                      m_TorchListOfConstantInts(dilationArray)))
      return rewriter.notifyMatchFailure(
          op, "Non-const dilation for pooling op unsupported.");
    // TOSA pooling only supports unit dilation.
    if (dilationArray[0] > 1 || dilationArray[1] > 1)
      return rewriter.notifyMatchFailure(
          op, "Cannot process non-unit pooling dilation.");

    if (failed(getOutputTypeAndPoolingParameters<AtenMaxPool2dOp,
                                                 tosa::MaxPool2dOp>(
            op, rewriter, adaptor.getSelf(), dilationArray, outputTy, kernel,
            stride, pad)))
      return rewriter.notifyMatchFailure(
          op, "invalid pooling parameters or input type");

    // Transpose to xHWC
    input = ConvertAtenPoolingBaseOp<AtenMaxPool2dOp, tosa::MaxPool2dOp>::
        transposePoolingInputToHwc(op, rewriter, adaptor.getSelf());

    return success();
  }
};

class ConvertAtenAvgPool2dOp
    : public ConvertAtenPoolingBaseOp<AtenAvgPool2dOp, tosa::AvgPool2dOp> {
public:
  using ConvertAtenPoolingBaseOp<AtenAvgPool2dOp,
                                 tosa::AvgPool2dOp>::ConvertAtenPoolingBaseOp;
  LogicalResult processInputs(AtenAvgPool2dOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              DenseI64ArrayAttr &kernel,
                              DenseI64ArrayAttr &stride, DenseI64ArrayAttr &pad,
                              Type &outputTy) const override {
    SmallVector<int64_t, 2> dilationArray{1, 1};
    if (failed(getOutputTypeAndPoolingParameters<AtenAvgPool2dOp,
                                                 tosa::AvgPool2dOp>(
            op, rewriter, adaptor.getSelf(), dilationArray, outputTy, kernel,
            stride, pad)))
      return rewriter.notifyMatchFailure(
          op, "invalid pooling parameters or input type");

    // Transpose to xHWC
    input = ConvertAtenPoolingBaseOp<AtenAvgPool2dOp, tosa::AvgPool2dOp>::
        transposePoolingInputToHwc(op, rewriter, adaptor.getSelf());

    return success();
  }
};

// Ref: Error checking based on the Torch to LinAlg lowering
template <typename AtenOpT, int fillVal>
class ConvertAtenConstPatternOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outType = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    if (!outType)
      return rewriter.notifyMatchFailure(op,
                                         "Only Tensor types supported in TOSA");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");

    // FIXME: Handle layout, device and pin_memory. Assume dtype has been
    // processed to set output type correctly?
    // The layout arg should be either `none` or `0` i.e. strided.
    if (!isa<Torch::NoneType>(op.getLayout().getType())) {
      int64_t tensorLayout;
      if (!matchPattern(op.getLayout(), m_TorchConstantInt(&tensorLayout)))
        return rewriter.notifyMatchFailure(
            op, "The layout arg should be either `none` or `0` i.e. strided.");
      else if (tensorLayout != torch_upstream::Layout::Strided)
        return rewriter.notifyMatchFailure(
            op, "The layout arg should be either `none` or `0` i.e. strided.");
    }

    bool pinMemory;
    if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
        (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return rewriter.notifyMatchFailure(
          op, "Unsupported pin_memory, should be either None or false");
    }

    SmallVector<int64_t> shape;
    if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(shape))) {
      return rewriter.notifyMatchFailure(
          op, "Shape must be a list of Scalar constants");
    }

    int64_t size = 1;
    for (auto s : shape)
      size *= s;

    SmallVector<int32_t> values(size, fillVal);
    auto constOp =
        tosa::getConstTensor<int32_t>(rewriter, op, values, shape).value();

    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, outType, constOp);

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenFillScalarOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outType = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    if (!outType || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "Only Tensor types with static shapes are currently supported");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");
    }
    Value constOp;
    if (failed(torchScalarToTosaTensor(
            rewriter, op, op.getValue(), constOp, outElemTy,
            makeShapeTorchCompatible(outType.getShape()))))
      return rewriter.notifyMatchFailure(
          op, "Supplied value must be a Scalar constant");

    auto newOp =
        rewriter.createOrFold<tosa::CastOp>(op.getLoc(), outType, constOp);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenMaskedFillOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outType = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));

    if (!outType || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "Only Tensor types with static shapes are currently supported");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");
    }

    // Not a tensor type.
    auto selfType = dyn_cast<TensorType>(adaptor.getSelf().getType());
    if (!selfType || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op,
          "Only tensor types with static shapes input are currently supported");

    auto maskType = dyn_cast<TensorType>(adaptor.getMask().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(
          op, "Only tensor types mask are currently supported");

    Value rhs = adaptor.getValue();
    auto rhsType = dyn_cast<TensorType>(rhs.getType());
    Value rhsAsTensor;
    if (!rhsType) { // scalar
      if (failed(torchScalarToTosaTensor(rewriter, op, op.getValue(),
                                         rhsAsTensor, rhs.getType(), {})))
        return rewriter.notifyMatchFailure(
            op, "Currently only scalar constants are supported for "
                "conversion in TOSA operation");
    } else { // tensor
      rhsType = dyn_cast<TensorType>(rhs.getType());
    }

    auto rhsTensor = rhsType ? rhs : rhsAsTensor;
    auto rhsTensorType = dyn_cast<TensorType>(rhsTensor.getType());
    if (rhsTensorType.getElementType() != outElemTy)
      rhsTensor = rewriter.create<tosa::CastOp>(
          op.getLoc(),
          RankedTensorType::get(rhsTensorType.getShape(), outElemTy),
          rhsTensor);

    rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, outType, adaptor.getMask(),
                                                rhsTensor, adaptor.getSelf());
    return success();
  }
};

// Legalizes the torch.clone op.
template <typename AtenOpT>
class ConvertAtenCloneOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t memoryFormat;
    if (!isa<Torch::NoneType>(op.getMemoryFormat().getType()) &&
        (!matchPattern(op.getMemoryFormat(),
                       m_TorchConstantInt(&memoryFormat)) ||
         (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
          memoryFormat != torch_upstream::MemoryFormat::ChannelsLast))) {
      return op.emitError(
          "unimplemented: only contiguous and channels last memory "
          "format is supported");
    }
    auto outType = dyn_cast<TensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()));
    rewriter.replaceOpWithNewOp<tosa::CastOp>(op, outType, adaptor.getSelf());

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenConstantPadNdOp>::matchAndRewrite(
    AtenConstantPadNdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value self = adaptor.getSelf();
  auto selfTy = cast<RankedTensorType>(self.getType());
  auto selfElemTy = selfTy.getElementType();
  int64_t rank = selfTy.getRank();

  // START the code snippet from
  // lib/Conversion/TorchToLinalg/TensorConstructors.cpp (see:
  // ConvertAtenConstantPadNdOp) Pattern match against the op's original
  // operands, because otherwise we will get the lowered version of the operands
  // which is harder to pattern match.
  SmallVector<int64_t> padInts;
  if (!matchPattern(op.getPad(), m_TorchListOfConstantInts(padInts)))
    return rewriter.notifyMatchFailure(op,
                                       "only support constant int pad ranges");
  uint64_t padRank = padInts.size() / 2;
  if (padRank * 2 != padInts.size())
    return rewriter.notifyMatchFailure(op, "pad range size is not even");
  if (rank < 0 || padRank > (uint64_t)rank)
    return rewriter.notifyMatchFailure(op, "padding exceeds tensor rank");

  // Initialize low/high paddings with 0 for all the dims.
  SmallVector<int64_t> lowPadding(/*Size=*/rank, /*Value=*/0);
  SmallVector<int64_t> highPadding(/*Size=*/rank, /*Value=*/0);
  // Add the requested padding - note op.pad() is highest dim first ordered
  // pairs of low,high.
  for (uint64_t i = 0; i < padRank; ++i) {
    lowPadding[rank - i - 1] = padInts[i * 2];
    highPadding[rank - i - 1] = padInts[i * 2 + 1];
  }
  // END the code snippet from
  // lib/Conversion/TorchToLinalg/TensorConstructors.cpp (see:
  // ConvertAtenConstantPadNdOp)

  llvm::SmallVector<int64_t> translatePadsList;

  for (unsigned int i = 0; i < rank; i++) {
    translatePadsList.push_back(lowPadding[i]);
    translatePadsList.push_back(highPadding[i]);
  }

  DenseElementsAttr paddingAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({rank, 2}, rewriter.getI64Type()),
      translatePadsList);

  Value padsList1 = rewriter.create<mlir::tosa::ConstOp>(
      loc, paddingAttr.getType(), paddingAttr);

  Value padValue = adaptor.getValue();
  Operation *padOp = padValue.getDefiningOp();
  padValue = padOp->getOperand(0);

  Value padTensor;
  if (failed(torchScalarToTosaTensor(rewriter, op.getOperation(), padValue,
                                     padTensor, selfElemTy, {})))
    return rewriter.notifyMatchFailure(
        op, "Pad value needs to be a scalar constant for conversion to "
            "TOSA pad operation");

  rewriter.replaceOpWithNewOp<mlir::tosa::PadOp>(
      op, getTypeConverter()->convertType(op.getType()), self, padsList1,
      padTensor);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenCatOp>::matchAndRewrite(
    AtenCatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  const TypeConverter *typeConverter = this->getTypeConverter();
  auto outType =
      cast<RankedTensorType>(typeConverter->convertType(op.getType()));
  int64_t rank = outType.getRank();
  int64_t dim;

  if (!outType || !outType.hasStaticShape()) {
    return rewriter.notifyMatchFailure(
        op, "Only Tensor types with static shapes are currently supported");
  }

  Location loc = op.getLoc();
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(op,
                                       "unimplemented: dim is not constant");
  }
  dim = toPositiveDim(dim, rank);
  if (!isValidDim(dim, rank)) {
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");
  }
  auto tensorList = op.getTensors();
  SmallVector<Value> tensorsTorchType;

  if (!getListConstructElements(tensorList, tensorsTorchType)) {
    return rewriter.notifyMatchFailure(
        op, "unimplemented: the tensor list is not from list construct");
  }
  auto builtinTensors =
      getTypeConvertedValues(rewriter, loc, typeConverter, tensorsTorchType);

  for (auto &in : builtinTensors)
    in = tosa::promoteType(rewriter, in, outType);

  auto result = tosa::CreateOpAndInfer<tosa::ConcatOp>(
      rewriter, loc, outType, builtinTensors, rewriter.getI32IntegerAttr(dim));
  rewriter.replaceOp(op, result.getResult());
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSqrtOp>::matchAndRewrite(
    AtenSqrtOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  // Converts AtenSqrtOp into pow(x, 0.5)
  auto self = adaptor.getSelf();
  auto selfTy = dyn_cast<TensorType>(self.getType());
  if (!selfTy)
    return rewriter.notifyMatchFailure(op,
                                       "Only Tensor types supported in TOSA");

  auto resultType =
      cast<RankedTensorType>(typeConverter->convertType(op.getType()));
  auto elementType = resultType.getElementType();

  if (isa<mlir::IntegerType>(selfTy.getElementType())) {
    self = rewriter.createOrFold<tosa::CastOp>(
        op->getLoc(), RankedTensorType::get(resultType.getShape(), elementType),
        self);
  }

  auto oneHalf =
      tosa::getConstTensor<float>(rewriter, op, 0.5, {}, elementType).value();

  rewriter.replaceOpWithNewOp<tosa::PowOp>(op, resultType, self, oneHalf);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenEmptyMemoryFormatOp>::matchAndRewrite(
    AtenEmptyMemoryFormatOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto loc = op.getLoc();
  MLIRContext *ctx = op->getContext();
  const TypeConverter *typeConverter = this->getTypeConverter();

  bool pinMemory;
  if (!isa<Torch::NoneType>(op.getPinMemory().getType()) &&
      (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
       pinMemory)) {
    return rewriter.notifyMatchFailure(
        op, "Unsupported pin_memory, should be either None or false");
  }

  if (!isa<Torch::NoneType>(op.getDevice().getType())) {
    std::string device;
    if (!matchPattern(op.getDevice(), m_TorchConstantDevice(device)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: device must be a constant str");
    if (device != "cpu")
      return rewriter.notifyMatchFailure(
          op, "unimplemented: device is expected to be none or cpu");
  }

  if (!isa<Torch::NoneType>(op.getLayout().getType())) {
    int64_t tensorLayout;
    if (!matchPattern(op.getLayout(), m_TorchConstantInt(&tensorLayout)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: layout must be a constant");
    if (tensorLayout != torch_upstream::Layout::Strided)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: layout is expected to be strided");
  }
  // Only `none`, `contiguous` and `preserve` memory_format are supported.
  if (!isa<Torch::NoneType>(op.getMemoryFormat().getType())) {
    int64_t memoryFormat;
    if (!matchPattern(op.getMemoryFormat(), m_TorchConstantInt(&memoryFormat)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: the memory format should be specified in "
              "an integer constant");
    if (memoryFormat != torch_upstream::MemoryFormat::Contiguous &&
        memoryFormat != torch_upstream::MemoryFormat::Preserve)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: only none, contiguous and preserve "
              "memory_format is supported");
  }

  SmallVector<Value> size;
  if (!getListConstructElements(op.getSize(), size))
    return rewriter.notifyMatchFailure(
        op, "unimplemented: size must be a ListConstruct");
  SmallVector<Value> resultSize =
      getTypeConvertedValues(rewriter, loc, typeConverter, size);
  auto resultType =
      cast<RankedTensorType>(typeConverter->convertType(op.getType()));

  DenseElementsAttr emptyVal;
  if (isa<Torch::NoneType>(op.getDtype().getType())) {
    emptyVal = DenseFPElementsAttr::get(resultType, {0.0F});
  } else {
    int64_t dtypeInt;
    if (!matchPattern(op.getDtype(), m_TorchConstantInt(&dtypeInt)))
      return rewriter.notifyMatchFailure(
          op, "unimplemented: dtype must be a constant integer or none");
    FailureOr<Type> maybeResultElementType =
        getTypeForScalarType(ctx, (torch_upstream::ScalarType)dtypeInt);
    if (failed(maybeResultElementType)) {
      return rewriter.notifyMatchFailure(
          op, "unable to convert `dtypeInt` to builtin type");
    }
    if (maybeResultElementType->isSignedInteger(64) ||
        maybeResultElementType->isIndex())
      emptyVal = DenseIntElementsAttr::get(resultType, {0L});
    if (maybeResultElementType->isSignedInteger(32))
      emptyVal = DenseIntElementsAttr::get(resultType, {0});
    else if (maybeResultElementType->isSignlessInteger(64))
      emptyVal = DenseIntElementsAttr::get(resultType, {0UL});
    else if (maybeResultElementType->isSignlessInteger(32))
      emptyVal = DenseIntElementsAttr::get(resultType, {0U});
    else if (maybeResultElementType->isSignedInteger(1) ||
             maybeResultElementType->isSignlessInteger(1))
      emptyVal = DenseIntElementsAttr::get(resultType, {false});
    else if (maybeResultElementType->isF64())
      emptyVal = DenseFPElementsAttr::get(resultType, {0.0});
    else if (maybeResultElementType->isF32())
      emptyVal = DenseFPElementsAttr::get(resultType, {0.0F});
    else
      return rewriter.notifyMatchFailure(
          op, "unsupported: dtype used for empty.memory_format is unsupported");
  }

  rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, resultType, emptyVal);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenRepeatInterleaveTensorOp>::matchAndRewrite(
    AtenRepeatInterleaveTensorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto outputTy =
      dyn_cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));
  if (!outputTy)
    return rewriter.notifyMatchFailure(
        op, "Only ranked tensor type outputs permitted");

  auto shape = outputTy.getShape();
  if (shape.size() != 1)
    return rewriter.notifyMatchFailure(op, "Only rank 1 tensors are permitted");

  int64_t outputSize;
  if (!matchPattern(op.getOutputSize(), m_TorchConstantInt(&outputSize))) {
    return rewriter.notifyMatchFailure(
        op, "Currently only scalar constants are supported for "
            "output_size in TOSA operation");
  }

  auto repeats = dyn_cast<tosa::ConstOp>(adaptor.getRepeats().getDefiningOp());
  if (!repeats)
    return rewriter.notifyMatchFailure(
        op, "Currently only constants are supported for "
            "repeats in TOSA operation");

  auto attr = repeats.getValue();
  if (!attr.isSplat())
    return rewriter.notifyMatchFailure(op, "Only single values are supported.");

  auto elementTy = outputTy.getElementType();
  if (!isa<mlir::IntegerType>(elementTy))
    return rewriter.notifyMatchFailure(op,
                                       "Only integer values are supported.");

  int64_t numberOfRepeats = attr.getSplatValue<llvm::APInt>().getSExtValue();

  // Create an array of repeated values
  auto createConstArrayOfRepeatedValues = [&](int64_t numOfRepeats) {
    SmallVector<int64_t> values;
    for (int64_t val = 0; val < outputSize / numberOfRepeats; ++val) {
      SmallVector<int64_t> newValues(numberOfRepeats, val);
      values.insert(values.end(), newValues.begin(), newValues.end());
    }
    return values;
  };

  auto newOp = tosa::getConstTensor<int64_t>(
      rewriter, op, createConstArrayOfRepeatedValues(numberOfRepeats), shape,
      elementTy);
  rewriter.replaceOp(op, *newOp);
  return success();
}

class SimplifyAtenIndexTensorWithSliceIndex
    : public OpRewritePattern<AtenIndexTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AtenIndexTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto outTy = dyn_cast<BaseTensorType>(op.getType());
    if (!outTy) {
      return rewriter.notifyMatchFailure(op, "requires tensor type");
    }

    SmallVector<Value> indices;
    if (!getListConstructElements(op.getIndices(), indices))
      return failure();

    TypedValue<BaseTensorType> input =
        dyn_cast<TypedValue<BaseTensorType>>(op.getSelf());
    if (!input) {
      return rewriter.notifyMatchFailure(op, "requires tensor type");
    }

    if (llvm::count_if(indices, [](Value v) {
          return !isa<Torch::NoneType>(v.getType());
        }) == 1) {
      return rewriter.notifyMatchFailure(op, "nothing to do");
    }

    auto loc = op->getLoc();

    for (size_t i = 0; i < indices.size(); ++i) {
      if (isa<Torch::NoneType>(indices[i].getType()))
        continue;

      auto indicesTy = dyn_cast<BaseTensorType>(indices[i].getType());
      if (!indicesTy || !indicesTy.areAllSizesKnown()) {
        return rewriter.notifyMatchFailure(
            op, "requires indices with static shape");
      }
      int64_t numIndices = std::accumulate(
          indicesTy.getSizes().begin(), indicesTy.getSizes().end(), 1,
          [&](int64_t a, int64_t b) { return a * b; });
      if (numIndices != 1)
        continue;

      auto inputTy = input.getType();
      SmallVector<int64_t> slicedShape{inputTy.getSizes()};
      slicedShape[i] = 1;
      auto slicedType =
          inputTy.getWithSizesAndDtype(slicedShape, inputTy.getDtype());

      auto none = rewriter.create<Torch::ConstantNoneOp>(op->getLoc());
      SmallVector<Value> sliceIndices{inputTy.getSizes().size(), none};
      sliceIndices[i] = reshapeTo(loc, rewriter, indices[i], {1});

      Value sliceIndicesV = rewriter.create<PrimListConstructOp>(
          loc, op.getIndices().getType(), sliceIndices);
      auto slicedInput = rewriter.create<AtenIndexTensorOp>(
          loc, slicedType, input, sliceIndicesV);

      SmallVector<int64_t> reshapedShape = slicedShape;
      reshapedShape.erase(reshapedShape.begin() + i);

      auto reshaped = reshapeTo(loc, rewriter, slicedInput, reshapedShape);

      SmallVector<Value> newIndicesList{indices};
      newIndicesList.erase(newIndicesList.begin() + i);

      Value newIndicesListV = rewriter.create<PrimListConstructOp>(
          loc, op.getIndices().getType(), newIndicesList);

      rewriter.replaceOpWithNewOp<AtenIndexTensorOp>(op, op.getType(), reshaped,
                                                     newIndicesListV);
      return success();
    }
    return failure();
  }
};

class SimplifyAtenIndexTensorWithNdIndex
    : public OpRewritePattern<AtenIndexTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AtenIndexTensorOp op,
                                PatternRewriter &rewriter) const override {
    auto outTy = dyn_cast<BaseTensorType>(op.getType());
    if (!outTy) {
      return rewriter.notifyMatchFailure(op, "requires tensor type");
    }

    SmallVector<Value> indices;
    if (!getListConstructElements(op.getIndices(), indices))
      return failure();

    TypedValue<BaseTensorType> input =
        dyn_cast<TypedValue<BaseTensorType>>(op.getSelf());
    if (!input) {
      return rewriter.notifyMatchFailure(op, "requires tensor type");
    }
    auto loc = op->getLoc();

    if (llvm::count_if(indices, [](Value v) {
          return !isa<Torch::NoneType>(v.getType());
        }) != 1) {
      return rewriter.notifyMatchFailure(op, "can only handle single None");
    }

    for (size_t i = 0; i < indices.size(); ++i) {
      if (isa<Torch::NoneType>(indices[i].getType()))
        continue;

      auto indicesTy = dyn_cast<BaseTensorType>(indices[i].getType());
      if (!indicesTy || !indicesTy.areAllSizesKnown()) {
        return rewriter.notifyMatchFailure(
            op, "requires indices with static shape");
      }
      if (indicesTy.getSizes().size() == 1) {
        continue;
      }

      // flatten indices
      int64_t numIndices = std::accumulate(
          indicesTy.getSizes().begin(), indicesTy.getSizes().end(), 1,
          [&](int64_t a, int64_t b) { return a * b; });

      auto newIndices =
          reshapeTo(op.getLoc(), rewriter, indices[i], {numIndices});

      SmallVector<Value> newIndicesList{indices};
      newIndicesList[i] = newIndices;

      Value newIndicesListV = rewriter.create<PrimListConstructOp>(
          loc, op.getIndices().getType(), newIndicesList);

      SmallVector<int64_t> indexOpShape{outTy.getSizes()};
      indexOpShape.erase(indexOpShape.begin() + i,
                         indexOpShape.begin() + i +
                             indicesTy.getSizes().size());
      indexOpShape.insert(indexOpShape.begin() + i, numIndices);

      auto indexOpType =
          outTy.getWithSizesAndDtype(indexOpShape, outTy.getOptionalDtype());
      auto indexed = rewriter.create<AtenIndexTensorOp>(loc, indexOpType, input,
                                                        newIndicesListV);

      auto reshaped = reshapeTo(loc, rewriter, indexed, outTy.getSizes());
      rewriter.replaceOp(op, reshaped);
      return success();
    }
    return failure();
  }
};
template <>
LogicalResult
ConvertAtenOp<Aten__InterpolateSizeListScaleListOp>::matchAndRewrite(
    Aten__InterpolateSizeListScaleListOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  // Converts torch.aten.__interpolate.size_list_scale_list to tosa.resize
  auto input = adaptor.getInput();
  auto inputTy = dyn_cast<RankedTensorType>(input.getType());
  if (!inputTy)
    return rewriter.notifyMatchFailure(op,
                                       "Only Tensor types supported in TOSA");
  auto inputRank = inputTy.getRank();
  if (inputRank != 4)
    return rewriter.notifyMatchFailure(op,
                                       "TOSA resize() takes rank==4 tensors.");

  auto inputShape = inputTy.getShape();
  auto inputElemTy = inputTy.getElementType();
  // TOSA works in NHWC. Perform the necessary transformations.
  std::optional<Value> nchwToNhwcTransposeConst =
      tosa::getConstTensor<int32_t>(rewriter, op,
                                    /*vec=*/{0, 2, 3, 1},
                                    /*shape=*/{static_cast<int32_t>(4)});
  SmallVector<int64_t> transposedInputShape(
      {inputShape[0], inputShape[2], inputShape[3], inputShape[1]});
  auto transposedInputTy = RankedTensorType::get(
      makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
  auto transposedInput =
      rewriter
          .create<tosa::TransposeOp>(
              op->getLoc(), getTypeConverter()->convertType(transposedInputTy),
              input, nchwToNhwcTransposeConst.value())
          .getResult();

  auto inputHeight = transposedInputShape[1];
  auto inputWidth = transposedInputShape[2];

  int outputHeight, outputWidth;
  if (!isa<Torch::NoneType>(op.getScaleFactor().getType())) {
    SmallVector<double, 2> scaleFactor;
    if (!matchPattern(op.getScaleFactor(),
                      m_TorchListOfConstantFloats(scaleFactor)))
      return rewriter.notifyMatchFailure(
          op, "non-const scale_factor parameter unsupported");

    outputHeight = inputHeight * scaleFactor[0];
    outputWidth = inputWidth * scaleFactor[1];

  } else {
    if (!isa<Torch::NoneType>(op.getSize().getType()))
      return rewriter.notifyMatchFailure(
          op, "Scale factor and size are both absent!");

    SmallVector<int64_t, 4> size;
    if (!matchPattern(op.getSize(), m_TorchListOfConstantInts(size)))
      return rewriter.notifyMatchFailure(
          op, "non-const size parameter unsupported");
    outputHeight = size[0];
    outputWidth = size[1];
  }

  std::string pyMode;
  if (!matchPattern(op.getMode(), m_TorchConstantStr(pyMode)))
    return rewriter.notifyMatchFailure(op,
                                       "non-const mode parameter unsupported");

  // All torch modes listed in
  // https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
  if (pyMode != "bilinear" && pyMode != "nearest")
    return rewriter.notifyMatchFailure(
        op, "Only nearest and bilinear interpolation modes supported");

  std::string mode;
  if (pyMode == "bilinear") {
    mode = "BILINEAR";
  } else {
    mode = "NEAREST_NEIGHBOR";
  }

  bool alignCorners;
  if (!matchPattern(op.getAlignCorners(), m_TorchConstantBool(&alignCorners)))
    return rewriter.notifyMatchFailure(
        op, "non-const align_corners parameter unsupported");

  bool recomputeScaleFactor;
  if (isa<Torch::NoneType>(op.getRecomputeScaleFactor().getType()))
    recomputeScaleFactor = false;
  else if (!matchPattern(op.getRecomputeScaleFactor(),
                         m_TorchConstantBool(&recomputeScaleFactor)))
    return rewriter.notifyMatchFailure(
        op, "non-const recompute_scale_factor parameter unsupported");
  if (recomputeScaleFactor)
    return rewriter.notifyMatchFailure(
        op, "Application of recompute_scale_factor not yet supported");

  bool antialias;
  if (!matchPattern(op.getAntialias(), m_TorchConstantBool(&antialias)))
    return rewriter.notifyMatchFailure(
        op, "non-const antialias parameter unsupported");
  if (antialias)
    return rewriter.notifyMatchFailure(
        op, "Application of antialias not yet supported");

  SmallVector<int64_t> transposedResizedOpShape(
      {inputShape[0], outputHeight, outputWidth, inputShape[1]});
  auto transposedResizedOpTy = RankedTensorType::get(
      makeShapeLLVMCompatible(transposedResizedOpShape), inputElemTy);

  // Formatting snake_case to match TOSA spec names for readability
  int scale_y_n, scale_y_d, offset_y, border_y;
  int scale_x_n, scale_x_d, offset_x, border_x;

  // Align corners sets the scaling ratio to (OH - 1)/(IH - 1)
  // rather than OH / IH. Similarly for width.
  auto normalize = [&](int input, int output, int &n, int &d, int &offset,
                       int &border) {
    // Dimension is length 1, we are just sampling from one value.
    if (input == 1) {
      n = output;
      d = 1;
      offset = 0;
      border = output - 1;
      return;
    }

    // Apply if aligned and capable to be aligned.
    bool apply_aligned = alignCorners && (output > 1);
    n = apply_aligned ? (output - 1) : output;
    d = apply_aligned ? (input - 1) : input;

    // Simplify the scalers, make sure they are even values.
    int gcd = std::gcd(n, d);
    n = 2 * n / gcd;
    d = 2 * d / gcd;

    offset = 0;

    // If nearest neighbours we need to guarantee we round up.
    if (mode == "NEAREST_NEIGHBOR" && alignCorners) {
      offset += n / 2;
    }

    // TBD: impact of antialias parameter here ?

    // We can compute this directly based on previous values.
    border = d * (output - 1) - n * (input - 1) + offset;
  };

  normalize(inputHeight, outputHeight, scale_y_n, scale_y_d, offset_y,
            border_y);
  normalize(inputWidth, outputWidth, scale_x_n, scale_x_d, offset_x, border_x);

  DenseI64ArrayAttr scale = rewriter.getDenseI64ArrayAttr(
      {scale_y_n, scale_y_d, scale_x_n, scale_x_d});
  DenseI64ArrayAttr offset =
      rewriter.getDenseI64ArrayAttr({offset_y, offset_x});
  DenseI64ArrayAttr border =
      rewriter.getDenseI64ArrayAttr({border_y, border_x});
  StringAttr modeAttr = rewriter.getStringAttr(mode);

  auto resizeOpResult =
      rewriter
          .create<tosa::ResizeOp>(op->getLoc(), transposedResizedOpTy,
                                  transposedInput, scale, offset, border,
                                  modeAttr)
          .getResult();

  auto resultType =
      cast<RankedTensorType>(typeConverter->convertType(op.getType()));
  std::optional<Value> nhwcToNchwTransposeConst =
      tosa::getConstTensor<int32_t>(rewriter, op,
                                    /*vec=*/{0, 3, 1, 2},
                                    /*shape=*/{static_cast<int32_t>(4)});
  // SmallVector<int64_t> transposedOutputShape(
  //     {transposedResizedOpShape[0], transposedResizedOpShape[3],
  //      transposedResizedOpShape[1], transposedResizedOpShape[2]});
  // auto transposedOutputType = RankedTensorType::get(
  //     makeShapeLLVMCompatible(transposedOutputShape), inputElemTy);
  rewriter
      .replaceOpWithNewOp<tosa::TransposeOp>(
          op, getTypeConverter()->convertType(resultType), resizeOpResult,
          nhwcToNchwTransposeConst.value())
      .getResult();

  return success();
}

} // namespace

// -----------------------------------------------------------------------------
// TorchToTosa Pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToTosa : public ConvertTorchToTosaBase<ConvertTorchToTosa> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<tosa::TosaDialect, tensor::TensorDialect,
                           arith::ArithDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    // The following ops are never the primary reason why lowering fails.
    // The backend contract only allows functions to return tensors thus there
    // is always another op using them.
    // When we have a chain of torch.constant.int followed by a unsupported
    // torch op, we want the pass to mention the unsupported torch op
    // in the error message.
    target.addLegalOp<ConstantNoneOp>();
    target.addLegalOp<ConstantBoolOp>();
    target.addLegalOp<ConstantIntOp>();
    target.addLegalOp<ConstantFloatOp>();
    target.addLegalOp<ConstantStrOp>();
    target.addLegalOp<ConstantDeviceOp>();
    target.addLegalOp<PrimListConstructOp>();
    target.addLegalOp<PrimTupleConstructOp>();
    target.addIllegalDialect<Torch::TorchDialect>();

    RewritePatternSet patterns(context);

    patterns.add<SimplifyAten_IndexPutImplOp>(context);
    patterns.add<SimplifyAten_IndexPutImplOpNone>(context);
    patterns.add<SimplifyAtenIndexTensorWithSliceIndex>(context);
    patterns.add<SimplifyAtenIndexTensorWithNdIndex>(context);
    patterns.add<ConvertAtenIndexTensorOpNone>(typeConverter, context);

#define INSERT_SIMPLIFY_OP_PATTERN(AtenOp)                                     \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<SimplifyAtenOp<AtenOp>>(typeConverter, context);
    INSERT_SIMPLIFY_OP_PATTERN(AtenConvolutionOp)
#undef INSERT_SIMPLIFY_OP_PATTERN

#define INSERT_UNARY_FPONLY_PATTERN(AtenOp, TosaOp)                            \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenUnaryFPOnlyOp<AtenOp, TosaOp>>(typeConverter,        \
                                                         context);
    INSERT_UNARY_FPONLY_PATTERN(AtenLogOp, tosa::LogOp)
    INSERT_UNARY_FPONLY_PATTERN(AtenExpOp, tosa::ExpOp)
#undef INSERT_UNARY_FPONLY_PATTERN

#define INSERT_UNARY_PATTERN(AtenOp, TosaOp)                                   \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenUnaryOp<AtenOp, TosaOp>>(typeConverter, context);
    INSERT_UNARY_PATTERN(AtenNegOp, tosa::NegateOp)
    INSERT_UNARY_PATTERN(AtenFloorOp, tosa::FloorOp)
    INSERT_UNARY_PATTERN(AtenRsqrtOp, tosa::RsqrtOp)
    INSERT_UNARY_PATTERN(AtenBitwiseNotOp, tosa::BitwiseNotOp)
    INSERT_UNARY_PATTERN(AtenCeilOp, tosa::CeilOp)
    INSERT_UNARY_PATTERN(AtenReciprocalOp, tosa::ReciprocalOp)
    INSERT_UNARY_PATTERN(AtenErfOp, tosa::ErfOp)
    INSERT_UNARY_PATTERN(AtenLogicalNotOp, tosa::LogicalNotOp)
    INSERT_UNARY_PATTERN(AtenCosOp, tosa::CosOp)
    INSERT_UNARY_PATTERN(AtenSinOp, tosa::SinOp)
#undef INSERT_UNARY_PATTERN

#define INSERT_BINARY_PATTERN(AtenOp, TosaOp)                                  \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenBinaryOp<AtenOp, TosaOp>>(typeConverter, context);
    INSERT_BINARY_PATTERN(AtenMaximumOp, tosa::MaximumOp)
    INSERT_BINARY_PATTERN(AtenMinimumOp, tosa::MinimumOp)
    INSERT_BINARY_PATTERN(AtenLogicalAndOp, tosa::LogicalAndOp)
    INSERT_BINARY_PATTERN(AtenLogicalOrOp, tosa::LogicalOrOp)
    INSERT_BINARY_PATTERN(AtenLogicalXorOp, tosa::LogicalXorOp)
#undef INSERT_BINARY_PATTERN

#define INSERT_BINARY_ADDSUB_PATTERN(AtenOp, TosaOp)                           \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAddSubOp<AtenOp, TosaOp>>(typeConverter, context);
    INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, tosa::AddOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, tosa::AddOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, tosa::SubOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenSubScalarOp, tosa::SubOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenRsubScalarOp, tosa::SubOp)
#undef INSERT_BINARY_ADDSUB_PATTERN

#define INSERT_BINARY_COMPARE_PATTERN(AtenOp, TosaOp)                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenCompareOp<AtenOp, TosaOp>>(typeConverter, context);
    INSERT_BINARY_COMPARE_PATTERN(AtenGtTensorOp, tosa::GreaterOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenGeScalarOp, tosa::GreaterEqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenGeTensorOp, tosa::GreaterEqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenLeScalarOp, tosa::GreaterEqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenLeTensorOp, tosa::GreaterEqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenGtScalarOp, tosa::GreaterOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenLtTensorOp, tosa::GreaterOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenLtScalarOp, tosa::GreaterOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenEqTensorOp, tosa::EqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenEqScalarOp, tosa::EqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenNeTensorOp, tosa::EqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenNeScalarOp, tosa::EqualOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenBitwiseAndTensorOp, tosa::BitwiseAndOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenBitwiseOrTensorOp, tosa::BitwiseOrOp)
    INSERT_BINARY_COMPARE_PATTERN(AtenBitwiseXorTensorOp, tosa::BitwiseXorOp)
#undef INSERT_BINARY_COMPARE_PATTERN

#define INSERT_BINARY_MUL_PATTERN(AtenOp)                                      \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMulOp<AtenOp>>(typeConverter, context);
    INSERT_BINARY_MUL_PATTERN(AtenMulTensorOp);
    INSERT_BINARY_MUL_PATTERN(AtenMulScalarOp);
#undef INSERT_BINARY_MUL_PATTERN

#define INSERT_BINARY_DIV_PATTERN(AtenOp)                                      \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenDivOp<AtenOp>>(typeConverter, context);
    INSERT_BINARY_DIV_PATTERN(AtenDivTensorOp);
    INSERT_BINARY_DIV_PATTERN(AtenDivScalarOp);
#undef INSERT_BINARY_DIV_PATTERN

#define INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenOp, ConversionFunc)              \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMultipleDimsReductionOp<AtenOp, ConversionFunc>>(    \
      typeConverter, context);
    INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenMeanDimOp,
                                      mlir::tosa::convertReduceMeanOp)
    INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenSumDimIntListOp,
                                      mlir::tosa::convertReduceSumOp)
    INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenLinalgVectorNormOp,
                                      mlir::tosa::convertLinalgVectorNormOp)
#undef INSERT_NDIMS_REDUCTION_OP_PATTERN

#define INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenOp, ConversionFunc)             \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOneDimReductionOp<AtenOp, ConversionFunc>>(          \
      typeConverter, context);
    INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenAnyDimOp,
                                       mlir::tosa::convertReduceAnyOp)
#undef INSERT_ONEDIM_REDUCTION_OP_PATTERN

#define INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenOp, ConversionFunc)            \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAllDimsReductionOp<AtenOp, ConversionFunc>>(         \
      typeConverter, context);
    INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenAllOp,
                                        mlir::tosa::convertReduceAllOp)
    INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenAnyOp,
                                        mlir::tosa::convertReduceAnyOp)
    INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenSumOp,
                                        mlir::tosa::convertReduceSumOp)
#undef INSERT_ALLDIMS_REDUCTION_OP_PATTERN

#define INSERT_SQUEEZE_OP_PATTERN(AtenOp, TemplateForm)                        \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<TemplateForm<AtenOp>>(typeConverter, context);
    INSERT_SQUEEZE_OP_PATTERN(AtenSqueezeOp, ConvertAtenSqueezeAllDimsOp)
    INSERT_SQUEEZE_OP_PATTERN(AtenSqueezeDimOp, ConvertAtenSqueezeOneDimOp)
#undef INSERT_SQUEEZE_OP_PATTERN

#define INSERT_MATMUL_ATENOP_PATTERN(AtenOp)                                   \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMatMulOp<AtenOp>>(typeConverter, context);
    INSERT_MATMUL_ATENOP_PATTERN(AtenMatmulOp);
#undef INSERT_MATMUL_ATEMOP_PATTERN

#define INSERT_MM_ATENOP_PATTERN(AtenOp)                                       \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMmOp<AtenOp>>(typeConverter, context);
    INSERT_MM_ATENOP_PATTERN(AtenMmOp);
    INSERT_MM_ATENOP_PATTERN(AtenBmmOp);
#undef INSERT_MM_ATEMOP_PATTERN

#define INSERT_LINEAR_ATENOP_PATTERN(AtenOp)                                   \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenLinearOp<AtenOp>>(typeConverter, context);
    INSERT_LINEAR_ATENOP_PATTERN(AtenLinearOp);
#undef INSERT_LINEAR_ATEMOP_PATTERN

#define INSERT_ADAPTIVE_POOLING_ATENOP_PATTERN(AtenOp, TosaOpT)                \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenAdaptivePoolingOp<AtenOp, TosaOpT>>(typeConverter,   \
                                                              context);
    INSERT_ADAPTIVE_POOLING_ATENOP_PATTERN(AtenAdaptiveAvgPool2dOp,
                                           tosa::AvgPool2dOp);
#undef INSERT_ADAPTIVE_POOLING_ATEMOP_PATTERN

    target.addIllegalOp<AtenMaxPool2dOp>();
    patterns.add<ConvertAtenMaxPool2dOp>(typeConverter, context);

    target.addIllegalOp<AtenAvgPool2dOp>();
    patterns.add<ConvertAtenAvgPool2dOp>(typeConverter, context);

#define INSERT_CONSTANT_FILL_PATTERN(AtenOp, fillVal)                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenConstPatternOp<AtenOp, fillVal>>(typeConverter,      \
                                                           context);
    INSERT_CONSTANT_FILL_PATTERN(AtenOnesOp, 1);
    INSERT_CONSTANT_FILL_PATTERN(AtenZerosOp, 0);
#undef INSERT_CONSTANT_FILL_PATTERN

#define INSERT_FILL_SCALAR_PATTERN(AtenOp)                                     \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenFillScalarOp<AtenOp>>(typeConverter, context);
    INSERT_FILL_SCALAR_PATTERN(AtenFill_ScalarOp);
    INSERT_FILL_SCALAR_PATTERN(AtenFillScalarOp);
#undef INSERT_FILL_SCALAR_PATTERN

#define INSERT_MASKED_FILL_PATTERN(AtenOp)                                     \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenMaskedFillOp<AtenOp>>(typeConverter, context);
    INSERT_MASKED_FILL_PATTERN(AtenMaskedFillScalarOp);
    INSERT_MASKED_FILL_PATTERN(AtenMaskedFillTensorOp);
#undef INSERT_MASKED_FILL_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp)                                          \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context);
    INSERT_ATENOP_PATTERN(AtenTanhOp);
    INSERT_ATENOP_PATTERN(AtenHardtanhBackwardOp);
    INSERT_ATENOP_PATTERN(AtenSigmoidOp);
    INSERT_ATENOP_PATTERN(AtenReluOp);
    INSERT_ATENOP_PATTERN(AtenLeakyReluOp);
    INSERT_ATENOP_PATTERN(AtenArgmaxOp);
    INSERT_ATENOP_PATTERN(AtenPowScalarOp);
    INSERT_ATENOP_PATTERN(AtenPowTensorScalarOp);
    INSERT_ATENOP_PATTERN(AtenPowTensorTensorOp);
    INSERT_ATENOP_PATTERN(AtenRsubScalarOp);
    INSERT_ATENOP_PATTERN(AtenConvolutionOp);
    INSERT_ATENOP_PATTERN(ValueTensorLiteralOp);
    INSERT_ATENOP_PATTERN(AtenReshapeOp);
    INSERT_ATENOP_PATTERN(AtenBatchNormOp);
    INSERT_ATENOP_PATTERN(AtenNativeLayerNormOp);
    INSERT_ATENOP_PATTERN(AtenFlattenUsingIntsOp);
    INSERT_ATENOP_PATTERN(AtenUnflattenIntOp);
    INSERT_ATENOP_PATTERN(AtenPermuteOp);
    INSERT_ATENOP_PATTERN(AtenLog2Op);
    INSERT_ATENOP_PATTERN(AtenThresholdOp);
    INSERT_ATENOP_PATTERN(AtenUnsqueezeOp);
    INSERT_ATENOP_PATTERN(AtenContiguousOp);
    INSERT_ATENOP_PATTERN(AtenDropoutOp);
    INSERT_ATENOP_PATTERN(AtenViewOp);
    INSERT_ATENOP_PATTERN(AtenGeluOp);
    INSERT_ATENOP_PATTERN(AtenGeluBackwardOp);
    INSERT_ATENOP_PATTERN(AtenEmbeddingOp);
    INSERT_ATENOP_PATTERN(AtenTransposeIntOp);
    INSERT_ATENOP_PATTERN(AtenMaxDimOp);
    INSERT_ATENOP_PATTERN(AtenSliceTensorOp);
    INSERT_ATENOP_PATTERN(AtenBroadcastToOp);
    INSERT_ATENOP_PATTERN(AtenGatherOp);
    INSERT_ATENOP_PATTERN(AtenIndexPutHackedTwinOp);
    INSERT_ATENOP_PATTERN(AtenIndexTensorHackedTwinOp);
    INSERT_ATENOP_PATTERN(AtenAbsOp);
    INSERT_ATENOP_PATTERN(AtenWhereSelfOp);
    INSERT_ATENOP_PATTERN(AtenClampOp);
    INSERT_ATENOP_PATTERN(AtenArangeStartStepOp);
    INSERT_ATENOP_PATTERN(PrimNumToTensorScalarOp);
    INSERT_ATENOP_PATTERN(AtenCopyOp);
    INSERT_ATENOP_PATTERN(AtenToDtypeOp);
    INSERT_ATENOP_PATTERN(AtenConstantPadNdOp);
    INSERT_ATENOP_PATTERN(AtenRemainderScalarOp);
    INSERT_ATENOP_PATTERN(AtenCatOp);
    INSERT_ATENOP_PATTERN(AtenSqrtOp);
    INSERT_ATENOP_PATTERN(AtenEmptyMemoryFormatOp);
    INSERT_ATENOP_PATTERN(AtenRepeatInterleaveTensorOp);
    INSERT_ATENOP_PATTERN(AtenIscloseOp);
    INSERT_ATENOP_PATTERN(Aten__InterpolateSizeListScaleListOp);
#undef INSERT_ATENOP_PATTERN

#define INSERT_CLONE_ATENOP_PATTERN(AtenOp)                                    \
  target.addIllegalOp<AtenOp>();                                               \
  patterns.add<ConvertAtenCloneOp<AtenOp>>(typeConverter, context);
    INSERT_CLONE_ATENOP_PATTERN(AtenCloneOp);
#undef INSERT_CLONE_ATENOP_PATTERN

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToTosaPass() {
  return std::make_unique<ConvertTorchToTosa>();
}
