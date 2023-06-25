// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitMatMul(in NodeProto op)
        {
            var (a, b) = GetInputExprs(op, 0, 1);
            var shapeA = IR.F.Tensors.ShapeOf(a);
            var shapeB = IR.F.Tensors.ShapeOf(b);
            var lhs = a;
            if (a.CheckedShape.Rank == 1)
            {
                var newShapeA = new Expr[] { 1L, shapeA[0] };
                lhs = IR.F.Tensors.Reshape(a, IR.F.Tensors.Stack(new IR.Tuple(newShapeA), 0));
            }

            var rhs = b;
            if (b.CheckedShape.Rank == 1)
            {
                var newShapeB = new Expr[] { shapeB[0], 1L };
                rhs = IR.F.Tensors.Reshape(b, IR.F.Tensors.Stack(new IR.Tuple(newShapeB), 0));
            }

            return F.Tensors.MatMul(lhs, rhs);
        }
    }
}
