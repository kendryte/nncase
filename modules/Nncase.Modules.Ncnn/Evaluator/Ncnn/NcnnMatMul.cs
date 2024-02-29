// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Ncnn;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnMatMul"/>.
/// </summary>
public class NcnnMatMulEvaluator : IEvaluator<NcnnMatMul>, ITypeInferencer<NcnnMatMul>, ICostEvaluator<NcnnMatMul>, IShapeEvaluator<NcnnMatMul>, IMetricEvaluator<NcnnMatMul>
{
    public static IRType VisitTensorType(TensorType lhs, TensorType rhs)
    {
        if (lhs.Shape.IsUnranked || rhs.Shape.IsUnranked)
        {
            return new TensorType(lhs.DType, Shape.Unranked);
        }

        // if (lhs.Shape[^1].IsUnknown || rhs.Shape[^2].IsUnknown)
        // {
        //     return new TensorType(lhs.DType, Shape.Unranked);
        // }
        if (lhs.DType != rhs.DType)
        {
            return new InvalidType("MatMul lhs and rhs have different DType");
        }

        if (lhs.Shape[^1] != rhs.Shape[^2] && lhs.Shape[^1] != Dimension.Unknown && rhs.Shape[^2] != Dimension.Unknown)
        {
            return new InvalidType("MatMul lhs and rhs have not compatiable shape");
        }

        if (lhs.Shape.Count == 2 && rhs.Shape.Count == 2)
        {
            return new TensorType(lhs.DType, new[] { lhs.Shape[0], rhs.Shape[1] });
        }

        var lhsShape = lhs.Shape.Rank >= rhs.Shape.Rank ? lhs.Shape.ToArray() : Enumerable.Repeat((Dimension)1, rhs.Shape.Rank - lhs.Shape.Rank).Concat(lhs.Shape).ToArray();
        var rhsShape = lhs.Shape.Rank <= rhs.Shape.Rank ? rhs.Shape.ToArray() : Enumerable.Repeat((Dimension)1, lhs.Shape.Rank - rhs.Shape.Rank).Concat(rhs.Shape).ToArray();

        var bigShape = Enumerable.Zip(lhsShape, rhsShape).SkipLast(2).Select(t =>
            t.First == Dimension.Unknown || t.Second == Dimension.Unknown
                ? Dimension.Unknown
                : System.Math.Max(t.First.FixedValue, t.Second.FixedValue)).ToArray();

        // batch and channel
        var front = bigShape;
        var end = new[] { lhs.Shape[^2], rhs.Shape[^1] };
        return new TensorType(lhs.DType, front.Concat(end).ToArray());
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnMatMul matMul)
    {
        var inputA = context.GetOrtArgumentValue(matMul, NcnnMatMul.InputA);
        var inputB = context.GetOrtArgumentValue(matMul, NcnnMatMul.InputB);

        // return OrtKI. (input, dim).ToValue();
        return OrtKI.MatMul(inputA, inputB).ToValue();
    }

    public IRType Visit(NcnnMatMul target, TensorType lhs, TensorType rhs)
    {
        return TypeInference.BroadcastType(lhs, rhs);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnMatMul target)
    {
        TensorType inputA, inputB;
        var dataType = target.LorR switch
        {
            0 => context.CheckArgumentType<TensorType>(target, NcnnMatMul.InputA).DType,
            1 => context.CheckArgumentType<TensorType>(target, NcnnMatMul.InputB).DType,
            2 => context.CheckArgumentType<TensorType>(target, NcnnMatMul.InputA).DType,
            _ => throw new NotSupportedException("never reach here"),
        };

        inputA = target.LorR switch
        {
            1 => new TensorType(dataType, target.ConstShape),
            _ => context.CheckArgumentType<TensorType>(target, NcnnMatMul.InputA),
        };

        inputB = target.LorR switch
        {
            2 => new TensorType(inputA.DType, target.ConstShape),
            _ => context.CheckArgumentType<TensorType>(target, NcnnMatMul.InputB),
        };

        return VisitTensorType(inputA, inputB);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnMatMul target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnMatMul target)
    {
        var lhs = context.GetArgumentType<TensorType>(target, NcnnMatMul.InputA);
        var rhs = context.GetArgumentType<TensorType>(target, NcnnMatMul.InputB);
        var outputType = context.GetReturnType<TensorType>();

        var k = (UInt128)lhs.Shape[^1].FixedValue;
        var m = MetricUtility.GetFLOPs(lhs) / k;
        var n = MetricUtility.GetFLOPs(rhs) / k;
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs) + CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = m * n * ((2 * k) - 1),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnMatMul target)
    {
        var lhs = context.GetArgumentShape(target, NcnnMatMul.InputA);
        var rhs = context.GetArgumentShape(target, NcnnMatMul.InputB);
        return ShapeExprUtility.BroadcastShape(lhs, rhs);
    }

    private Expr Visit(TensorType inputA, TensorType inputB)
    {
        return Cast(IR.F.ShapeExpr.MatMulShape(inputA.Shape, inputB.Shape), DataTypes.Int32);
    }
}
