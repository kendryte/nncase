// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitWhere(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var (x, y) = GetInputExprs(op, 1, 2);
            return Where(input, x, y);
        }
    }
}
