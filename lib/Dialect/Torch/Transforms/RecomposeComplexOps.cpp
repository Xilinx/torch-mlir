//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class RecomposeSliceCopy_ : public OpRewritePattern<AtenCopy_Op> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenCopy_Op op,
                                PatternRewriter &rewriter) const override {
    // This pattern replaces the in-place mutation of a slice of a tensor with
    // an `index_put` op. Since the slice of the tensor can have a different
    // shape than the full tensor, this pattern requires the `copy_` op to not
    // have users to avoid mismached types. This restriction can be removed by
    // inserting another slice after the `index_put` that creates a tensor of
    // the same shape as the operand to `copy_`.
    if (!op.use_empty())
      return rewriter.notifyMatchFailure(
          op, "`AtenCopy_Op` must not have any users");
    if (!op.getSelf().getDefiningOp() ||
        !isa<AtenSliceTensorOp>(op.getSelf().getDefiningOp()))
      return rewriter.notifyMatchFailure(
          op, "defining op is not `AtenSliceTensorOp`");
    auto sliceOp = cast<AtenSliceTensorOp>(op.getSelf().getDefiningOp());

    // Get indices
    int64_t dim;
    if (!matchPattern(sliceOp.getDim(), m_TorchConstantInt(&dim)))
      return failure();
    int64_t end;
    if (!matchPattern(sliceOp.getEnd(), m_TorchConstantInt(&end)))
      return failure();

    Value newEnd = sliceOp.getEnd();
    Value dimSize = rewriter.create<AtenSizeIntOp>(
        op.getLoc(), sliceOp.getSelf(), sliceOp.getDim());
    if (end < 0) {
      newEnd =
          rewriter.create<AtenAddIntOp>(op.getLoc(), dimSize, sliceOp.getEnd());
    }
    newEnd = rewriter.create<PrimMinIntOp>(op.getLoc(), newEnd, dimSize);

    Value noneVal = rewriter.create<ConstantNoneOp>(op.getLoc());
    Value falseVal = rewriter.create<ConstantBoolOp>(op.getLoc(), false);

    // Create IndexPut_Op
    BaseTensorType tensorType = op.getType().cast<BaseTensorType>();
    Type rangeType = tensorType.getWithSizesAndDtype(
        {kUnknownSize}, tensorType.getOptionalDtype());
    Value range = rewriter.create<AtenArangeStartStepOp>(
        op.getLoc(), rangeType, sliceOp.getStart(), newEnd, sliceOp.getStep(),
        /*dtype=*/noneVal, /*layout=*/noneVal, /*device=*/noneVal,
        /*pin_memory=*/noneVal);

    SmallVector<Value> indicesVector;
    for (auto i = 0; i < dim; i++)
      indicesVector.push_back(noneVal);
    indicesVector.push_back(range);
    Type indicesType = tensorType.getWithSizesAndDtype(
        /*optionalSizes=*/std::nullopt, /*optionalDtype=*/nullptr);
    Value indices = rewriter.create<PrimListConstructOp>(
        op.getLoc(),
        Torch::ListType::get(op->getContext(),
                             Torch::OptionalType::get(indicesType)),
        indicesVector);

    Value sliceOpInput = sliceOp.getSelf();
    rewriter.replaceOpWithNewOp<Aten_IndexPutImpl_Op>(
        op, sliceOpInput.getType(), sliceOpInput, indices, op.getSrc(),
        /*accumulate=*/falseVal, /*unsafe=*/falseVal);

    if (sliceOp->use_empty())
      rewriter.eraseOp(sliceOp);

    return success();
  }
};

class RecomposeSelectFill_ : public OpRewritePattern<AtenFill_TensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenFill_TensorOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getSelf().getDefiningOp() ||
        !isa<AtenSelectIntOp>(op.getSelf().getDefiningOp()))
      return failure();
    auto selectOp = cast<AtenSelectIntOp>(op.getSelf().getDefiningOp());

    // Get indices
    int64_t dim;
    if (!matchPattern(selectOp.getDim(), m_TorchConstantInt(&dim)))
      return failure();

    Value noneVal = rewriter.create<ConstantNoneOp>(op.getLoc());
    Value falseVal = rewriter.create<ConstantBoolOp>(op.getLoc(), false);

    // Create IndexPut_Op
    // Convert indexNum to indexTensor for the selectOp
    BaseTensorType selectOutTy =
        selectOp.getType().template cast<BaseTensorType>();
    SmallVector<int64_t> empty;
    auto dtype = getTypeForTorchType(selectOp.getContext(),
                                     selectOp.getIndex().getType());
    Type emptyTensorType =
        selectOutTy.getWithSizesAndDtype(llvm::ArrayRef(empty), dtype);
    Value indexTensor = rewriter.create<PrimNumToTensorScalarOp>(
        selectOp.getLoc(), emptyTensorType, selectOp.getIndex());

    // Create indicesVector for IndexPut_Op by TorchNone and indexTensor
    BaseTensorType tensorType = op->getResultTypes()[0].cast<BaseTensorType>();
    SmallVector<Value> indicesVector(dim - 1, noneVal);
    indicesVector.push_back(indexTensor);

    Value indices = rewriter.create<PrimListConstructOp>(
        op.getLoc(),
        Torch::ListType::get(op->getContext(),
                             Torch::OptionalType::get(tensorType)),
        indicesVector);

    rewriter.replaceOpWithNewOp<Aten_IndexPutImpl_Op>(
        op, op->getResultTypes(), selectOp.getSelf(), indices, op.getValue(),
        /*accumulate=*/falseVal, /*unsafe=*/falseVal);

    return success();
  }
};

class RecomposeUnbindListUnpack : public OpRewritePattern<PrimListUnpackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimListUnpackOp op,
                                PatternRewriter &rewriter) const override {
    // recompose AtenUnbindOp + PrimListUnpackOp to select.int
    auto unbind = dyn_cast<AtenUnbindIntOp>(op.getOperand().getDefiningOp());
    if (!unbind)
      return rewriter.notifyMatchFailure(op, "Input is not AtenUnbindIntOp");
    if (isListPotentiallyMutated(unbind.getResult()))
      return rewriter.notifyMatchFailure(
          op, "AtenUnbindIntOp result is potentially mutated");
    Value dim = unbind.getDim();
    Value input = unbind.getSelf();
    SmallVector<Value> slices;
    for (size_t i = 0; i < op.getNumResults(); i++) {
      // rewrite to select.int op
      auto resultTy = op.getResult(i).getType();
      auto index = rewriter.create<Torch::ConstantIntOp>(
          op->getLoc(), rewriter.getI64IntegerAttr(i));
      auto newSelect = rewriter.create<AtenSelectIntOp>(op->getLoc(), resultTy,
                                                        input, dim, index);
      slices.push_back(newSelect);
    }
    rewriter.replaceOp(op, slices);
    if (unbind.getResult().use_empty())
      rewriter.eraseOp(unbind);
    return success();
  }
};

class RecomposeUnbindGetItem : public OpRewritePattern<Aten__Getitem__TOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten__Getitem__TOp op,
                                PatternRewriter &rewriter) const override {
    // recompose AtenUnbindIntOp + __getitem__t to select.int
    auto unbind = dyn_cast<AtenUnbindIntOp>(op.getList().getDefiningOp());
    if (!unbind)
      return rewriter.notifyMatchFailure(op, "Input is not AtenUnbindIntOp");
    if (isListPotentiallyMutated(unbind.getResult()))
      return rewriter.notifyMatchFailure(
          op, "AtenUnbindIntOp result is potentially mutated");
    int64_t index;
    if (!matchPattern(op.getIdx(), m_TorchConstantInt(&index)))
      return rewriter.notifyMatchFailure(
          op, "Expected `idx` of `Aten__Getitem__TOp` to be a constant int");

    Location loc = op.getLoc();
    Value dim = unbind.getDim();
    Value input = unbind.getSelf();
    // rewrite to slice op
    auto resultTy = op.getResult().getType();
    Value newSelect = rewriter.create<AtenSelectIntOp>(loc, resultTy, input,
                                                       dim, op.getIdx());
    rewriter.replaceOp(op, newSelect);
    if (unbind.getResult().use_empty())
      rewriter.eraseOp(unbind);
    return success();
  }
};

class RecomposeSplitTensorPrimListUnpackOp : public OpRewritePattern<PrimListUnpackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimListUnpackOp op,
                                PatternRewriter &rewriter) const override {

    auto torchList = op.getOperand();
    if (isListPotentiallyMutated(torchList))
      return failure();

    auto split = torchList.getDefiningOp<Torch::AtenSplitTensorOp>();
    if (!split)
        return failure();
    int64_t size = 0;
    if (!matchPattern(split.getSplitSize(), m_TorchConstantInt(&size)))
      return failure();

    Value constOne = rewriter.create<Torch::ConstantIntOp>(
        op->getLoc(), rewriter.getI64IntegerAttr(1));
    std::vector<Value> results;
    int64_t start = 0;

    for (size_t i = 0; i < op->getNumResults(); ++i) {
      results.push_back(rewriter.create<AtenSliceTensorOp>(
          op->getLoc(),
          op.getResult(i).getType(),
          split.getSelf(),
          /*dim=*/split.getDim(),
          /*start=*/
          rewriter.create<Torch::ConstantIntOp>(
              op->getLoc(), rewriter.getI64IntegerAttr(start)),
          /*end=*/
          rewriter.create<Torch::ConstantIntOp>(
              op->getLoc(), rewriter.getI64IntegerAttr(start + size)),
          /*step=*/constOne));
      start += size;
    }
    rewriter.replaceOp(op, results);
    if (split->use_empty())
      rewriter.eraseOp(split);

    return success();
  }
};

class RecomposeSplitTensorGetItemOp
    : public OpRewritePattern<Aten__Getitem__TOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Aten__Getitem__TOp op,
                                PatternRewriter &rewriter) const override {
    // recompose AtenSplitTensorOp + __getitem__t to AtenSliceTensorOp
    auto splitTensorOp =
        dyn_cast<AtenSplitTensorOp>(op.getList().getDefiningOp());
    if (!splitTensorOp)
      return rewriter.notifyMatchFailure(op, "Input is not AtenSplitTensorOp");
    if (isListPotentiallyMutated(splitTensorOp.getResult()))
      return rewriter.notifyMatchFailure(
          op, "SplitTensorOp result is potentially mutated");
    int64_t index;
    if (!matchPattern(op.getIdx(), m_TorchConstantInt(&index)))
      return rewriter.notifyMatchFailure(
          op, "Expected `idx` of `Aten__Getitem__TOp` to be a constant int");

    int64_t splitSize;
    if (!matchPattern(splitTensorOp.getSplitSize(),
                      m_TorchConstantInt(&splitSize)))
      return rewriter.notifyMatchFailure(
          op,
          "Expected `SplitSize` of `AtenSplitTensorOp` to be a constant int");

    Location loc = op.getLoc();
    Value step =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value start = rewriter.create<ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(index * splitSize));
    Value end = rewriter.create<ConstantIntOp>(
        loc, rewriter.getI64IntegerAttr(index * splitSize + splitSize));
    Value sliceTensorOp = rewriter.create<AtenSliceTensorOp>(
        loc, op.getResult().getType(), splitTensorOp.getSelf(),
        splitTensorOp.getDim(), start, end, step);
    rewriter.replaceOp(op, sliceTensorOp);
    if (splitTensorOp.getResult().use_empty())
      rewriter.eraseOp(splitTensorOp);
    return success();
  }
};

class RecomposeChunkListUnpack : public OpRewritePattern<PrimListUnpackOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimListUnpackOp op,
                                PatternRewriter &rewriter) const override {
    // recompose AtenChunkOp + PrimListUnpackOp to AtenSliceTensorOps
    auto chunk = dyn_cast<AtenChunkOp>(op.getOperand().getDefiningOp());
    if (!chunk)
      return rewriter.notifyMatchFailure(op, "Input is not AtenChunkOp");
    if (isListPotentiallyMutated(chunk.getResult()))
      return rewriter.notifyMatchFailure(
          op, "AtenChunkOp result is potentially mutated");
    Value dim = chunk.getDim();
    Value input = chunk.getSelf();
    Value chunks = chunk.getChunks();
    Location loc = chunk.getLoc();
    Value totalSize = rewriter.create<Torch::AtenSizeIntOp>(loc, input, dim);

    // chunkSize = floordiv(totalSize + chunks - 1, chunks)
    Value cstOne =
        rewriter.create<ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
    Value dividend = rewriter.create<AtenAddIntOp>(loc, totalSize, chunks);
    dividend = rewriter.create<AtenSubIntOp>(loc, dividend, cstOne);
    Value chunkSize = rewriter.create<AtenFloordivIntOp>(loc, dividend, chunks);

    SmallVector<Value> slices;
    for (size_t i = 0; i < op.getNumResults(); i++) {
      // rewrite to slice op with
      // start = chunkSize * i,
      // end = lastIndex ? totalSize : chunkSize * (i+1)
      auto resultTy = op.getResult(i).getType();
      auto index = rewriter.create<Torch::ConstantIntOp>(
          op->getLoc(), rewriter.getI64IntegerAttr(i));
      auto start = rewriter.create<AtenMulIntOp>(loc, index, chunkSize);
      Value end;
      if (i == op.getNumResults() - 1) {
        end = totalSize;
      } else {
        auto nextIdx = rewriter.create<AtenAddIntOp>(loc, index, cstOne);
        end = rewriter.create<AtenMulIntOp>(loc, nextIdx, chunkSize);
      }
      Value sliceTensorOp = rewriter.create<AtenSliceTensorOp>(
          loc, resultTy, input, dim, start, end, cstOne);
      slices.push_back(sliceTensorOp);
    }
    rewriter.replaceOp(op, slices);
    // erase chunkOp if no user left
    if (chunk.getResult().use_empty())
      rewriter.eraseOp(chunk);
    return success();
  }
};
} // namespace

namespace {
class RecomposeComplexOpsPass
    : public RecomposeComplexOpsBase<RecomposeComplexOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // pattern.add calls go here
    patterns.add<RecomposeSliceCopy_>(context);
    patterns.add<RecomposeSelectFill_>(context);
    patterns.add<RecomposeSplitTensorGetItemOp>(context);
    patterns.add<RecomposeUnbindListUnpack>(context);
    patterns.add<RecomposeUnbindGetItem>(context);
    patterns.add<RecomposeSplitTensorPrimListUnpackOp>(context);
    patterns.add<RecomposeChunkListUnpack>(context);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createRecomposeComplexOpsPass() {
  return std::make_unique<RecomposeComplexOpsPass>();
}
