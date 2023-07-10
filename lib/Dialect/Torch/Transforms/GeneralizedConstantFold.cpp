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
#include <set>
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
  ReplaceConstantOp(MLIRContext *context,  std::set<std::string>& seenOps)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context), seenOps(seenOps) {}
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
      llvm::errs() << "Constant folding this op:" << op->getName() << "\n";
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

      if (seenOps.insert(op->getName().getStringRef().str()).second) {
        #if 0
        op->dump();
        op->getOperand(0).dump();
        // Compute the number of users
        auto users = op->getOperand(0).getUsers();
        llvm::errs() << "users: " << std::distance(users.begin(), users.end()) << "\n";
        #endif
        if(seenOps.size() == 1)
          llvm::errs() << "Constant folding this op:" << op->getName() << "\n";
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
   
  std::set<std::string> &seenOps;
};

class AddChains : public mlir::OpRewritePattern<tosa::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(tosa::AddOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto inputAdd = op.getInput1().getDefiningOp<tosa::AddOp>();
    auto inputConst = op.getInput2().getDefiningOp<tosa::ConstOp>();
    if (inputAdd && inputConst) {
      auto inputConst2 = inputAdd.getInput2().getDefiningOp<tosa::ConstOp>();
      if (inputConst2) {
        // TODO: This is not mathematically correct, but we don't care about
        // numerical values in this exercise
        rewriter.replaceOpWithNewOp<tosa::AddOp>(
            op, op.getType(), inputAdd.getInput1(), inputConst.getResult());
        return success();
      }
    }
    return failure();
  }
};

class MulAddChains : public mlir::OpRewritePattern<tosa::MulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(tosa::MulOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto inputAdd = op.getInput1().getDefiningOp<tosa::AddOp>();
    auto inputConst = op.getInput2().getDefiningOp<tosa::ConstOp>();
    if (inputAdd && inputConst) {
      auto inputConst2 = inputAdd.getInput2().getDefiningOp<tosa::ConstOp>();
      if (inputConst2) {
        // TODO: This is not mathematically correct, but we don't care about
        // numerical values in this exercise
        rewriter.replaceOpWithNewOp<tosa::AddOp>(
            op, op.getType(), inputAdd.getInput1(), inputConst.getResult());
        return success();
      }
    }
    return failure();
  }
};

template <typename OpType>
class FoldIntoArg : public mlir::OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(OpType op, mlir::PatternRewriter &rewriter) const override {

    if constexpr(std::is_same_v<OpType, tosa::CustomOp>) {
      if (op.getIdentifier() != "layer_norm")
        return failure();
    }

    SmallVector<Value> arguments;
    for(auto operand: op->getOperands()) {
      if (!isConstantLike(operand)) {
        auto custom = llvm::dyn_cast_if_present<tosa::CustomOp>(
        operand.getDefiningOp());

        if (custom && custom.getIdentifier() == "arg") {
          // push all operands of custom into arguments
          for(auto customOperand: custom->getOperands()) {
            arguments.push_back(customOperand);
          }
        } else if(isa<BlockArgument>(operand)) {
          arguments.push_back(operand);
        } else {
          return failure();
        }
      }
    }

    if(arguments.empty()) 
      return failure();

    auto *ctx = op->getContext();
    auto identifier = StringAttr::get(ctx, "arg");
    auto implementAttr = StringAttr::get(ctx, "custom");
    auto config = StringAttr::get(ctx, "UNDEF");

    rewriter
        .replaceOpWithNewOp<tosa::CustomOp>(op, op->getResult(0).getType(), identifier,
                                            config, implementAttr,
                                            SmallVector<Value>{arguments})
        .getResult(0);
    return success();
  }
};

struct GeneralizedConstantFoldPass
    : public GeneralizedConstantFoldBase<GeneralizedConstantFoldPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    std::set<std::string> seenOps;
    patterns.add<ReplaceConstantOp>(context, seenOps);
    patterns.add<AddChains>(context);
    patterns.add<FoldIntoArg<tosa::EqualOp>>(context);
    patterns.add<FoldIntoArg<tosa::LogicalNotOp>>(context);
    patterns.add<FoldIntoArg<tosa::CastOp>>(context);
    patterns.add<FoldIntoArg<tosa::AddOp>>(context);
    patterns.add<FoldIntoArg<tosa::SubOp>>(context);
    patterns.add<FoldIntoArg<tosa::MulOp>>(context);
    patterns.add<FoldIntoArg<tosa::PowOp>>(context);
    patterns.add<FoldIntoArg<tosa::ReciprocalOp>>(context);
    patterns.add<FoldIntoArg<tosa::ReduceSumOp>>(context);
    patterns.add<FoldIntoArg<tosa::GreaterOp>>(context);
    patterns.add<FoldIntoArg<tosa::ReshapeOp>>(context);
    patterns.add<FoldIntoArg<tosa::TransposeOp>>(context);
    patterns.add<FoldIntoArg<tosa::GatherOp>>(context);
    patterns.add<FoldIntoArg<tosa::SelectOp>>(context);
    patterns.add<FoldIntoArg<tosa::SliceOp>>(context);
    patterns.add<FoldIntoArg<tosa::ScatterOp>>(context);
    patterns.add<FoldIntoArg<tosa::CustomOp>>(context);
    patterns.add<FoldIntoArg<tosa::ConcatOp>>(context);
    patterns.add<FoldIntoArg<tosa::ArgMaxOp>>(context);
    patterns.add<FoldIntoArg<tosa::PadOp>>(context);

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
