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
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "TorchF64toF32"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

/// Rewrite an op with its new operands.
class ChangeType : public ConversionPattern {
public:
  ChangeType(TypeConverter &typeConverter, MLIRContext *context,
             PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, Pattern::MatchAnyOpTypeTag{}, benefit,
                          context) {}

  LogicalResult
  matchAndRewriteLiteral(ValueTensorLiteralOp op, ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) const {

    // Tensors with integer types need to be converted to signless integer
    // element type. All tensors with element types other than integer can reuse
    // existing elements attribute.
    // TODO: what about unsigned integer?
    if (auto elements = op.getValueAttr().dyn_cast<DenseFPElementsAttr>()) {
      if (elements.getElementType().isF64()) {
        assert(elements.isSplat());
        auto v = elements.getSplatValue<double>();
        auto valueAttr = DenseFPElementsAttr::get(
            elements.getType().cloneWith({},
                                         mlir::FloatType::getF32(getContext())),
            (float)v);

        auto outputTy = getTypeConverter()->convertType(op.getType());
        rewriter.replaceOpWithNewOp<ValueTensorLiteralOp>(op, outputTy,
                                                          valueAttr);
        return success();
      }
    }

    return failure();
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    if (op->getDialect()->getNamespace() != "torch")
      return failure();

    if (auto vtl = dyn_cast<ValueTensorLiteralOp>(op))
      return matchAndRewriteLiteral(vtl, operands, rewriter);
    
    llvm::SmallVector<Type, 4> resultTys;
    if (getTypeConverter()
            ->convertTypes(op->getResultTypes(), resultTys)
            .failed())
      llvm_unreachable("Cannot convert types");



    OperationState newProducerOpState(op->getLoc(),
                                      op->getName().getStringRef(), operands,
                                      resultTys, op->getAttrs());
    Operation *newOp = rewriter.create(newProducerOpState);

    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }
};

Value getDtypeIntValueForType(OpBuilder &builder, Location loc, Type dtype) {
  int intType = (int)getScalarTypeForType(dtype);
  return builder.create<ConstantIntOp>(loc, builder.getI64IntegerAttr(intType));
}

bool isValueTensorF64(Type t) {
  auto vtt = t.dyn_cast<ValueTensorType>();
  return vtt && vtt.hasDtype() && vtt.getDtype().isF64();
}

class TorchF64toF32 : public F64toF32Base<TorchF64toF32> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    // convert !torch.vtensor<[],f64> to !torch.vtensor<[],f32>
    typeConverter.addConversion(
        [context](ValueTensorType type) -> Optional<Type> {
          if (!type.hasDtype())
            return {};
          if (!type.hasSizes())
            return {};

          if (auto floatType = type.getDtype().dyn_cast<mlir::FloatType>()) {
            if (!floatType.isF64())
              return {};

            return ValueTensorType::get(context, type.getOptionalSizes(),
                                        mlir::FloatType::getF32(context));
          }
          return {};
        });

    typeConverter.addTargetMaterialization(
        [](OpBuilder &builder, ValueTensorType type, ValueRange inputs,
           Location loc) -> Value {
          Value none = builder.create<ConstantNoneOp>(loc);
          Value cstFalse = builder.create<ConstantBoolOp>(loc, false);
          
          return builder.createOrFold<AtenToDtypeOp>(
              loc, type, inputs[0],
              getDtypeIntValueForType(builder, loc, type.getDtype()),
              /*non_blocking=*/cstFalse, /*copy=*/cstFalse,
              /*memory_format=*/none);
        });
    auto sourceMaterialization = [](OpBuilder &builder,
                                    Torch::ValueTensorType type,
                                    ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1);
      Value none = builder.create<ConstantNoneOp>(loc);
      Value cstFalse = builder.create<ConstantBoolOp>(loc, false);
      return builder.createOrFold<AtenToDtypeOp>(
          loc, type, inputs[0],
          getDtypeIntValueForType(builder, loc, type.getDtype()),
          /*non_blocking=*/cstFalse, /*copy=*/cstFalse, /*memory_format=*/none);
    };
    typeConverter.addSourceMaterialization(sourceMaterialization);
    typeConverter.addArgumentMaterialization(sourceMaterialization);

    RewritePatternSet patterns(context);

    //target.addLegalOp<AtenToDtypeOp>();
    target.addDynamicallyLegalDialect<Torch::TorchDialect>([](Operation *op) {
      return !any_of(op->getResultTypes(), isValueTensorF64) &&
             !any_of(op->getOperandTypes(), isValueTensorF64);
    });

    patterns.add<ChangeType>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createTorchF64toF32Pass() {
  return std::make_unique<TorchF64toF32>();
}
