// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitSoftmax(in NodeProto op)
        {
            return GetOpSet(op) < 13
                ? SoftmaxV1(op)
                : SoftmaxV13(op);
        }

        private Expr SoftmaxV1Process(in NodeProto op, Func<Expr, Expr, Expr> f)
        {
            var input = GetSingleInputExpr(op);
            var axis = (int)GetIntAttribute(op, "axis", 1);
            var inShape = ShapeOf(input);
            Expr axisExpr = axis < 0
                ? axis + Cast(Rank(input), DataTypes.Int32)
                : Tensor.From<int>(new[] { axis });
            var first = Prod(Slice(inShape, new[] { 0 }, axisExpr, 1));
            var second = Prod(Slice(inShape, axisExpr, Rank(input), 1));
            var beforeShape = Stack(new IR.Tuple(first, second), 0);
            var afterShape = ShapeOf(input);
            return Reshape(
                f(
                    Reshape(input, beforeShape),
                    1),
                afterShape);
        }

        private Expr SoftmaxV13Process(in NodeProto op, Func<Expr, Expr, Expr> f)
        {
            var input = GetSingleInputExpr(op);
            var axis = GetIntAttribute(op, "axis", -1);
            return f(input, IR.F.Math.Select(axis < 0, (Rank(input) + axis)[0], axis));
        }

        private Expr SoftmaxV1(in NodeProto op)
        {
            return SoftmaxV1Process(op, Softmax);
        }

        private Expr SoftmaxV13(in NodeProto op)
        {
            return SoftmaxV13Process(op, Softmax);
        }

        private Expr VisitLogSoftmax(in NodeProto op)
        {
            return GetOpSet(op) < 13
                ? LogSoftmaxV1(op)
                : LogSoftmaxV13(op);
        }

        private Expr LogSoftmaxV1(in NodeProto op)
        {
            return SoftmaxV1Process(op, LogSoftmax);
        }

        private Expr LogSoftmaxV13(in NodeProto op)
        {
            return SoftmaxV13Process(op, LogSoftmax);
        }

        private Expr VisitSoftplus(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            return Softplus(input);
        }

        private Expr VisitSoftsign(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            return Softsign(input);
        }
    }
}
