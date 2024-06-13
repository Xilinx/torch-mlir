//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchOnnxToTorch/Patterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using llvm::dbgs;
using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::onnx_c;

#define DEBUG_TYPE "torch-onnx"

LogicalResult OnnxCustomOpConversionPattern::matchAndRewrite(
    Torch::OperatorOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto foundIt = namedHandlers.find(op.getNameAttr());
  if (foundIt == namedHandlers.end())
    return failure();
  // If the op has an individual version, use it.
  int64_t opDomainVersion = domainVersion;
  if (auto attr =
          op->getAttrOfType<IntegerAttr>("torch.onnx_meta.opset_version")) {
    if (auto type = dyn_cast<IntegerType>(attr.getType())) {
      if (type && type.isSigned()) {
        opDomainVersion =
            op->getAttrOfType<IntegerAttr>("torch.onnx_meta.opset_version")
                .getSInt();
      }
    }
  }
  auto &reggies = foundIt->second;
  for (const HandlerReg &reg : reggies) {
    if (opDomainVersion < reg.sinceVersion) {
      LLVM_DEBUG(dbgs() << ": skipping conversion " << foundIt->first
                        << ", sinceVersion=" << reg.sinceVersion
                        << ", for domainVersion=" << domainVersion
                        << ", opDomainVersion=" << opDomainVersion << "\n");
      continue;
    }
    if (succeeded(reg.callback(OpBinder(op), rewriter))) {
      return success();
    } else {
      LLVM_DEBUG(dbgs() << ": conversion failed to apply: " << foundIt->first
                        << ", sinceVersion=" << reg.sinceVersion << "\n");
    }
  }
  return rewriter.notifyMatchFailure(op, "no matching versioned converter");
}

void OnnxCustomOpConversionPattern::populateLegalizedNames(
    DenseSet<StringAttr> &legalizedNames) {
  for (auto it : namedHandlers)
    legalizedNames.insert(it.first);
}

void OnnxCustomOpConversionPattern::onOp(StringRef name, int64_t sinceVersion,
                                         HandlerFn callback) {
  SmallString<64> fullName(domainPrefix);
  fullName.append(name);
  StringAttr nameAttr = StringAttr::get(getContext(), fullName);
  namedHandlers[nameAttr].push_back(HandlerReg(callback, sinceVersion));
}
