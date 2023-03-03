//===- LowerToBackendContract.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "LowerQuantization"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {
class LowerAtenQuantizePerTensorOp
    : public OpConversionPattern<AtenQuantizePerTensorOp> {
public:
  using OpConversionPattern<AtenQuantizePerTensorOp>::OpConversionPattern;
  using OpAdaptor = typename AtenQuantizePerTensorOp::Adaptor;
  LogicalResult
  matchAndRewrite(AtenQuantizePerTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto stype = op.self().getType().cast<ValueTensorType>();
    auto dtype = getTypeConverter()
                     ->convertType(op->getResult(0).getType())
                     .cast<ValueTensorType>();

    auto mult = rewriter.create<Torch::AtenDivScalarOp>(
        op->getLoc(), stype, op.self(), adaptor.scale());

    Value one = rewriter.create<Torch::ConstantIntOp>(
        op->getLoc(), rewriter.getI64IntegerAttr(1));
    auto add = rewriter.create<Torch::AtenAddScalarOp>(
        op->getLoc(), stype, mult, adaptor.zero_point(), one);

    auto newop = rewriter.createOrFold<TorchConversion::ToIntOp>(
        op->getLoc(), dtype, add);

    rewriter.replaceOp(op, newop);

    return success();
  }
};

class LowerAtenIntReprOp
    : public OpConversionPattern<AtenIntReprOp> {
public:
  using OpConversionPattern<AtenIntReprOp>::OpConversionPattern;
  using OpAdaptor = typename AtenIntReprOp::Adaptor;
  LogicalResult
  matchAndRewrite(AtenIntReprOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    
    rewriter.replaceOp(op, adaptor.self());
    return success();
  }
};


class LowerQuantization : public LowerQuantizationBase<LowerQuantization> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
        [](Torch::ValueTensorType type) -> Optional<Type> {
          if (!type.getDtype().dyn_cast_or_null<Torch::QInt8Type>()) {
            return type;
          }

          auto *context = type.getContext();
          auto elementTy = IntegerType::get(context, 8, IntegerType::Signed);
          return type.getWithSizesAndDtype(type.getSizes(), elementTy);
        });


    RewritePatternSet patterns(context);
    target.addLegalDialect<Torch::TorchDialect>();
    target.addLegalOp<TorchConversion::ToIntOp>();
    target.addIllegalOp<AtenQuantizePerTensorOp>();
    target.addIllegalOp<AtenIntReprOp>();

    patterns.add<LowerAtenQuantizePerTensorOp>(typeConverter, context);
    patterns.add<LowerAtenIntReprOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::TorchConversion::createLowerQuantizationPass() {
  return std::make_unique<LowerQuantization>();
}