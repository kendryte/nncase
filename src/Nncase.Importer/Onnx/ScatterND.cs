// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitScatterND(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var indices = GetInputExpr(op, 1);
            var updates = GetInputExpr(op, 2);
            return F.Tensors.ScatterND(input, indices, updates);
        }
    }
}
