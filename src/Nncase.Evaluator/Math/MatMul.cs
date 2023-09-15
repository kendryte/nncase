// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using MatMul = Nncase.IR.Math.MatMul;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="MatMul"/>.
/// </summary>
public class MatMulEvaluator : IEvaluator<MatMul>, ITypeInferencer<MatMul>, ICostEvaluator<MatMul>, IShapeEvaluator<MatMul>, IMetricEvaluator<MatMul>
{
    public static IRType VisitDistributedType(DistributedType a, DistributedType b)
    {
        if (VisitTensorType(a.TensorType, b.TensorType) is not TensorType outType)
        {
            return new InvalidType(string.Empty);
        }

        if (a.Placement != b.Placement)
        {
            return new InvalidType("placement not equal");
        }

        var aRank = a.TensorType.Shape.Rank;
        var bRank = a.TensorType.Shape.Rank;

        var ndsbp = new SBP[a.Placement.Rank];
        for (int i = 0; i < a.Placement.Rank; i++)
        {
            var invalid = new InvalidType($"({a.NdSbp[i]}, {b.NdSbp[i]}) not support");
            switch (a.NdSbp[i], b.NdSbp[i])
            {
                // split on k
                case (SBPSplit { Axis: int ax }, SBPSplit { Axis: int bx }):
                    if (ax == (aRank - 1) && bx == (bRank - 2))
                    {
                        ndsbp[i] = SBP.P;
                    }
                    else if ((ax == (aRank - 1) && bx != (bRank - 2)) || (ax != (aRank - 1) && bx == (bRank - 2)))
                    {
                        return invalid;
                    }
                    else
                    {
                        if (ax == bx)
                        {
                            ndsbp[i] = SBP.S(ax);
                        }
                        else
                        {
                            return invalid;
                        }
                    }

                    break;
                case (SBPSplit { Axis: int ax }, SBPBroadCast):
                    if (ax == aRank - 1)
                    {
                        return invalid;
                    }

                    ndsbp[i] = SBP.S(ax);
                    break;
                case (SBPBroadCast, SBPSplit { Axis: int bx }):
                    if (bx == bRank - 2)
                    {
                        return invalid;
                    }

                    ndsbp[i] = SBP.S(bx);
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(outType, ndsbp, a.Placement);
    }

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
    public IValue Visit(IEvaluateContext context, MatMul matMul)
    {
        var input = context.GetOrtArgumentValue(matMul, MatMul.Lhs);
        var other = context.GetOrtArgumentValue(matMul, MatMul.Rhs);
        return OrtKI.MatMul(input, other).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, MatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, MatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, MatMul.Rhs);
        return (lhs, rhs) switch
        {
            (DistributedType a, DistributedType b) => VisitDistributedType(a, b),
            (TensorType a, TensorType b) => VisitTensorType(a, b),
            _ => new InvalidType(string.Empty),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, MatMul target)
    {
        var lhs = context.GetArgumentType<IRType>(target, MatMul.Lhs);
        var rhs = context.GetArgumentType<IRType>(target, MatMul.Rhs);
        var outputType = context.GetReturnType<IRType>();

        uint macPerElement = 1;
        if (lhs is TensorType { Shape: Shape lhsShape })
        {
            macPerElement = lhsShape[^1].IsFixed ? (uint)lhsShape[^1].FixedValue : 1U;
        }
        else if (lhs is DistributedType distributedType)
        {
            var lhsType = DistributedUtilities.GetDividedTensorType(distributedType, out var _);
            macPerElement = lhsType.Shape[^1].IsFixed ? (uint)lhsType.Shape[^1].FixedValue : 1U;
        }

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, macPerElement),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, MatMul target)
    {
        var lhs = context.GetArgumentType<TensorType>(target, MatMul.Lhs);
        var rhs = context.GetArgumentType<TensorType>(target, MatMul.Rhs);
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

    public Expr Visit(IShapeEvaluateContext context, MatMul target)
    {
        var lhs = context.GetArgumentShape(target, MatMul.Lhs);
        var rhs = context.GetArgumentShape(target, MatMul.Rhs);
        return Cast(IR.F.ShapeExpr.MatMulShape(lhs, rhs), DataTypes.Int32);
    }
}
