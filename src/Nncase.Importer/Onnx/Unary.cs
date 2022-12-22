// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Math;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitUnary(NodeProto op, UnaryOp unaryOp)
        {
            var input = GetInputExpr(op, 0);
            return F.Math.Unary(unaryOp, input);
        }
    }
}
