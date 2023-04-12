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
        private Expr VisitReduce(in NodeProto op, ReduceOp reduceOp, float initValue)
        {
            return ReduceCore(op, reduceOp, initValue, expr => expr);
        }

        private Expr ReduceCore(in NodeProto op, ReduceOp reduceOp, float initValue, Func<Expr, Expr> f)
        {
            var input = GetInputExpr(op, 0);
            var axis = GetAxesAttribute(op, input);
            var keepDims = GetBoolAttribute(op, "keepdims", true);
            return F.Tensors.Reduce(reduceOp, f(input), axis, initValue, keepDims);
        }

        private Expr ReduceSumZero(in NodeProto op, Func<Expr, Expr> f)
        {
            return ReduceCore(op, ReduceOp.Sum, 0f, f);
        }

        private Expr VisitReduceL1(in NodeProto op)
        {
            return ReduceSumZero(op, F.Math.Abs);
        }

        private Expr VisitReduceL2(in NodeProto op)
        {
            return F.Math.Sqrt(
                ReduceSumZero(op, F.Math.Square));
        }

        // ReduceLogSum(x) = Log(ReduceSum(x))
        private Expr VisitReduceLogSum(in NodeProto op)
        {
            return F.Math.Log(
                ReduceSumZero(op, expr => expr));
        }

        // ReduceLogSumExp(x) = Log(Sum(Exp(x)))
        private Expr VisitReduceLogSumExp(in NodeProto op)
        {
            return F.Math.Log(
                ReduceSumZero(op, F.Math.Exp));
        }

        // ReduceSumSquare(x) = Sum(Square(x))
        private Expr VisitReduceSumSquare(in NodeProto op)
        {
            return ReduceSumZero(op, F.Math.Square);
        }
    }
}
