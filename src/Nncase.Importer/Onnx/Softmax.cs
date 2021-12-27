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
        private Expr VisitSoftmax(in NodeProto op)
        {
            return GetOpSet(op) < 13
                ? SoftmaxV1(op)
                : SoftmaxV13(op);
        }

        private Expr SoftmaxV1(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            var axis = GetIntAttribute(op, "axis", -1);
            var inShape = F.Tensors.ShapeOp(input);
            Expr axisExpr = axis < 0
                ? axis + F.Tensors.Rank(input)
                : axis;
            // todo:axis < 0?
            var first = F.Tensors.Size(F.Tensors.Slice(inShape, new[] {0}, new[] {(int)axis}, 1));
            var second = F.Tensors.Size(F.Tensors.Slice(inShape, new[] {(int)axis}, F.Tensors.Rank(input) - 1, 1));
            var beforeShape = F.Tensors.Concat(new IR.Tuple(first, second), 0);
            var afterShape = F.Tensors.ShapeOp(input);
            return F.Tensors.Reshape(
                F.NN.SoftMax(
                    F.Tensors.Reshape(input, beforeShape),
                    axis),
                afterShape);
        }

        private Expr SoftmaxV13(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            var axis = GetIntAttribute(op, "axis", -1);
            return F.NN.SoftMax(input, axis);
        }

        private Expr VisitLogSoftmax(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var axis = GetIntAttribute(op, "axis", -1);
            return F.NN.LogSoftMax(input, axis);
        }

        private Expr VisitSoftplus(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            return F.NN.SoftPlus(input);
        }

        private Expr VisitSoftsign(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            return F.NN.SoftSign(input);
        }
    }
}