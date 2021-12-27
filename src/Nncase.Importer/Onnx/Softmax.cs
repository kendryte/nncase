// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;

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
            var inShape = ShapeOp(input);
            Expr axisExpr = axis < 0
                ? axis + Rank(input)
                : Const.FromSpan<int>(new[] {axis});
            var first = Prod(Slice(inShape, new[] {0}, axisExpr, 1));
            var second = Prod(Slice(inShape, axisExpr, Rank(input) , 1));
            var beforeShape = Concat(new IR.Tuple(first, second), 0);
            var afterShape = ShapeOp(input);
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
            return f(input, axis);
        }
        
        private Expr SoftmaxV1(in NodeProto op)
        {
            return SoftmaxV1Process(op, SoftMax);
        }

        private Expr SoftmaxV13(in NodeProto op)
        {
            return SoftmaxV13Process(op, SoftMax);
        }

        private Expr VisitLogSoftmax(in NodeProto op)
        {
            return GetOpSet(op) < 13
                ? LogSoftmaxV1(op)
                : LogSoftmaxV13(op);
        }

        private Expr LogSoftmaxV1(in NodeProto op)
        {
            return SoftmaxV1Process(op, LogSoftMax);
        }

        private Expr LogSoftmaxV13(in NodeProto op)
        {
            return SoftmaxV13Process(op, LogSoftMax);
        }
        
        private Expr VisitSoftplus(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            return SoftPlus(input);
        }

        private Expr VisitSoftsign(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            return SoftSign(input);
        }
    }
}