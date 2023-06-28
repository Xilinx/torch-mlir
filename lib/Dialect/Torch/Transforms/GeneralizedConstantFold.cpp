//===- ReduceOpVariants.cpp --------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

bool isConstantLike(Value v) {
  Attribute attr;
  return matchPattern(v, m_Constant(&attr));
}

class ReplaceConstantOp : public RewritePattern {
public:
  ReplaceConstantOp(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    if(op->getNumResults() != 0 && op->getUsers().empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    if (op->getNumOperands() == 0) {
      // typical for ops like tosa.const
      return failure();
    }

    if (!llvm::all_of(op->getOperands(),
                      [](Value v) { return isConstantLike(v); })) {
      return failure();
    }

    if (op->getNumResults() == 0) {
      llvm::errs() <<"unhandled zero-result op in constant folding\n";
      return failure();
    }

    if (op->getNumResults() > 1) {
      llvm::errs() << "unhandled multi-result op in constant folding\n";
      return failure();
    }

    auto resultTy = op->getResultTypes()[0];
    if (auto torchDialect = dyn_cast<TorchDialect>(op->getDialect())) {
      auto ret = torchDialect->materializeConstant(rewriter, Attribute(),
                                                   resultTy, op->getLoc());
      if (!ret) {
        resultTy.dump();
         llvm::errs() << "unhandled type in constant folding torch\n";
        return failure();
      }
      rewriter.replaceOp(op, ret->getResult(0));
      return success();
    } else if(auto tosaDialect = dyn_cast<tosa::TosaDialect>(op->getDialect())) {

      auto shapeTy = dyn_cast<ShapedType>(resultTy);
      if(!shapeTy) {
         llvm::errs() << "non-shape type in constant folding tosa\n";
        return failure();
      }
      Dialect* dialect = getContext()->getLoadedDialect<BuiltinDialect>();
      auto interf = dyn_cast<OpAsmDialectInterface>(dialect);
      FailureOr<AsmDialectResourceHandle> rawHandle = interf->declareResource("__elided__");
      if (failed(rawHandle)) {
         llvm::errs() << "unknown 'resource' key\n";
        return failure();
      }
      auto *handle = dyn_cast<DenseResourceElementsHandle>(&*rawHandle);
      if (!handle) {
         llvm::errs() << "invalid `dense_resource` handle type\n";
        return failure();
      }

      rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, op->getResultTypes()[0],
                                         DenseResourceElementsAttr::get(shapeTy, *handle));
      return success(); 
    }

    op->dump();
    for(auto in: op->getOperands()) {
      in.dump();
    }
    op->emitError("unhandled dialect in constant folding");
    exit(0);
    return failure();
  }
};

struct GeneralizedConstantFoldPass
    : public GeneralizedConstantFoldBase<GeneralizedConstantFoldPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<ReplaceConstantOp>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createGeneralizedConstantFoldPass() {
  return std::make_unique<GeneralizedConstantFoldPass>();
}
