// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
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

        private Expr SoftmaxV1Process(in NodeProto op, Func<Expr, Dimension, Expr> f)
        {
            var input = GetSingleInputExpr<Expr>(op);
            var rank = input.CheckedShape.Rank;
            var axis = (int)Dimension.Positive(GetIntAttribute(op, "axis", 1), rank).FixedValue;
            var inShape = (RankedShape)ShapeOf(input).AsShape();
            var first = TensorUtilities.GetProduct(inShape[..axis]);
            var second = TensorUtilities.GetProduct(inShape[axis..]);
            var beforeShape = new RankedShape(first, second);
            return Reshape(
                f(
                    Reshape(input, beforeShape),
                    1L),
                inShape);
        }

        private Expr SoftmaxV13Process(in NodeProto op, Func<Expr, Dimension, Expr> f)
        {
            var input = GetSingleInputExpr<Expr>(op);
            var axis = GetIntAttribute(op, "axis", -1);
            return f(input, Dimension.Positive(axis, Rank(input).AsDim()));
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
            var input = GetSingleInputExpr<Expr>(op);
            return Softplus(input);
        }

        private Expr VisitSoftsign(in NodeProto op)
        {
            var input = GetSingleInputExpr<Expr>(op);
            return Softsign(input);
        }
    }
}
