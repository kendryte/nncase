// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitRange(in NodeProto op)
        {
            var (start, limit) = GetInputExprs(op, 0, 1);
            var delta = GetInputExpr(op, 2);

            // todo:for float?
            return F.Tensors.Range(start, limit, delta);
        }
    }
}
