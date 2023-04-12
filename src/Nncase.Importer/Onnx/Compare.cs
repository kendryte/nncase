// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Math;
using Onnx;
using static Nncase.IR.F.Math;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitCompare(in NodeProto op, CompareOp compareOp)
        {
            var (lhs, rhs) = GetInputExprs(op, 0, 1);
            return Compare(compareOp, lhs, rhs);
        }
    }
}
