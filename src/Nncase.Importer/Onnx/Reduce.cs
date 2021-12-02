// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitReduce(in NodeProto op, ReduceOp reduceOp, float initValue)
        {
            var input = GetInputExpr(op, 0);
            var axis = Const.FromSpan<long>(GetAxisAttribute(op, "axes"));
            var keepDims = GetBoolAttribute(op, "keepdims", true);
            return F.Tensors.Reduce(reduceOp, input, axis, initValue, keepDims);
        }
    }
}