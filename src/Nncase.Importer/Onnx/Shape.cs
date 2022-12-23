// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitShape(in NodeProto op)
        {
            var opSet = GetOpSet(op);
            if (opSet < 15)
            {
                return ShapeV13(op);
            }
            else
            {
                return ShapeV15(op);
            }
        }

        private Expr ShapeV13(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return F.Tensors.ShapeOf(input);
        }

        private Expr ShapeV15(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var start = GetIntAttribute(op, "start", 0);
            var inShape = F.Tensors.ShapeOf(input);
            var end = GetOptionIntAttribute(op, "end");
            Expr endValue = end ? new[] { end.Value() } : F.Tensors.ShapeOf(inShape);
            return F.Tensors.Slice(
                inShape,
                new[] { start },
                endValue,
                1);
        }
    }
}
