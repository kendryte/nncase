// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.ShapeExpr;
using Nncase.Utilities;
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator.ShapeExpr;

[EvaluatorGenerator]
[TypeInferGenerator]
public partial class MatMulShapeEvaluator : IEvaluator<MatMulShape>, ITypeInferencer<MatMulShape>, ICostEvaluator<MatMulShape>, IShapeEvaluator<MatMulShape>, IMetricEvaluator<MatMulShape>
{
    public IValue Visit(Tensor lhs, Tensor rhs)
    {
        // var lhsRank = lhs.Shape.Rank;
        // var rhsRank = rhs.Shape.Rank;
        // var lhsShape = lhs.ToArray<int>();
        // var rhsShape = rhs.ToArray<int>();
        //
        // Expr newLhs, newRhs;
        // Expr front;
        // if (lhsRank == rhsRank)
        // {
        //     newLhs = ShapeExprUtility.Slice(lhsShape, 0, lhsRank - 2);
        //     newRhs = ShapeExprUtility.Slice(rhsShape, 0, rhsRank - 2);
        //     front = IR.F.Math.Max(newLhs, newRhs);
        // }
        // else if (lhsRank > rhsRank)
        // {
        //     newLhs = ShapeExprUtility.Slice(lhsShape, 0, lhsRank - 2);
        //     front = newLhs;
        // }
        // else
        // {
        //     newLhs = Enumerable.Repeat(1, rhsRank - lhsRank).ToArray();
        //     front = newLhs;
        // }
        //
        // var end = Stack(new IR.Tuple(lhsShape[lhsRank - 2], rhsShape[rhsRank - 1]), 0);
        // return Concat(new IR.Tuple(front, end), 0).Evaluate();
        var lhsShape = lhs.ToArray<int>();
        var rhsShape = rhs.ToArray<int>();
        var newLhs = To4D(lhsShape);
        var newRhs = To4D(rhsShape);
        var bigShapeSize = System.Math.Max(lhsShape.Length, rhsShape.Length);
        var newShape = new List<int>();
        for (int i = 0; i < bigShapeSize - 2; i++)
        {
            newShape.Add(System.Math.Max(newLhs[i + 4 - bigShapeSize], newRhs[i + 4 - bigShapeSize]));
        }

        newShape.Add(lhsShape[^2]);
        newShape.Add(rhsShape[^1]);
        return Value.FromTensor(newShape.Select(x => (long)x).ToArray());
    }

    public IRType Visit(TensorType lhs, TensorType rhs)
    {
        var shape = new[] { lhs, rhs }.MaxBy(tt => tt.Shape.Rank)!.Shape;
        return new TensorType(DataTypes.Int64, shape);
    }

    public Cost Visit(ICostEvaluateContext context, MatMulShape target)
    {
        return CostUtility.GetShapeExprCost();
    }

    public Expr Visit(IShapeEvaluateContext context, MatMulShape target)
    {
        // todo: broadcast
        var lhsRank = context.GetArgumentShape(target, MatMulShape.Lhs);
        var rhsRank = context.GetArgumentShape(target, MatMulShape.Rhs);
        return IR.F.Math.Max(lhsRank, rhsRank);
    }

    public Metric Visit(IMetricEvaluateContext context, MatMulShape target)
    {
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    private int[] To4D(int[] shape)
    {
        return Enumerable.Repeat(0, 4 - shape.Length).Concat(shape).ToArray();
    }
}
