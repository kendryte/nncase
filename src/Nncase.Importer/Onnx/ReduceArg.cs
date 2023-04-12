// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitReduceArg(in NodeProto op, ReduceArgOp reduceArgOp)
        {
            // op version > 11 have select_last_index
            var input = GetInputExpr(op, 0);
            var axis = GetIntAttribute(op, "axis", 0);
            var keepDims = GetBoolAttribute(op, "keepdims", true);
            var selectLastIndex = GetBoolAttribute(op, "select_last_index", false);
            return F.Tensors.ReduceArg(reduceArgOp, DataTypes.Int64, input, axis, keepDims, selectLastIndex);
        }
    }
}
