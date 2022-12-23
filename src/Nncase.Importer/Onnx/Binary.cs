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
        private Expr VisitBinary(NodeProto op, BinaryOp binaryOp)
        {
            var (lhs, rhs) = GetInputExprs(op, 0, 1);
            return F.Math.Binary(binaryOp, lhs, rhs);
        }
    }
}
