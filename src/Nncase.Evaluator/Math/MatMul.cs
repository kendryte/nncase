﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
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
    public static IRType VisitDistributedType(DistributedType a, DistributedType b, bool packingK = false, DimInfo? dimInfo = null)
    {
        if (VisitTensorType(a.TensorType, b.TensorType, packingK, dimInfo) is not TensorType outType)
        {
            return new InvalidType($"{a.TensorType} {b.TensorType} not support");
        }

        if (a.Placement != b.Placement)
        {
            return new InvalidType("placement not equal");
        }

        var aRank = a.TensorType.Shape.Rank;
        var bRank = b.TensorType.Shape.Rank;
        var oRank = outType.Shape.Rank;
        var aPad = oRank - aRank;
        var bPad = oRank - bRank;
        var (lm, lk, rk, _) = dimInfo ?? new(aRank - 2, aRank - 1, bRank - 2, bRank - 1);

        var ndsbp = new SBP[a.Placement.Rank];
        for (int i = 0; i < a.Placement.Rank; i++)
        {
            var invalid = new InvalidType($"({a.NdSBP[i]}, {b.NdSBP[i]}) not support");
            switch (a.NdSBP[i], b.NdSBP[i])
            {
                // split on k
                case (SBPSplit { Axis: int ax }, SBPSplit { Axis: int bx }):
                    if (ax == lk && bx == rk)
                    {
                        ndsbp[i] = SBP.P;
                    }
                    else if ((ax == lk && bx != rk) || (ax != lk && bx == rk))
                    {
                        return invalid;
                    }
                    else
                    {
                        if ((ax + aPad) == (bx + bPad))
                        {
                            ndsbp[i] = SBP.S(ax + aPad);
                        }
                        else
                        {
                            return invalid;
                        }
                    }

                    break;
                case (SBPSplit { Axis: int ax }, SBPBroadCast):
                    if (ax == lk)
                    {
                        return invalid;
                    }

                    // invalid (S, B) if B is not broacast matmul
                    if (ax < lm && !(bRank <= 2 || (ax + aPad - bPad >= 0 && b.TensorType.Shape[ax + aPad - bPad] == 1)))
                    {
                        return invalid;
                    }

                    ndsbp[i] = SBP.S(ax + aPad);
                    break;
                case (SBPBroadCast, SBPSplit { Axis: int bx }):
                    if (bx == rk)
                    {
                        return invalid;
                    }

                    // invalid (B, S) if A is not broacast matmul
                    if (bx < rk && !(aRank <= 2 || (bx + bPad - aPad >= 0 && a.TensorType.Shape[bx + bPad - aPad] == 1)))
                    {
                        return invalid;
                    }

                    ndsbp[i] = SBP.S(bx + bPad);
                    break;
                case (SBPBroadCast, SBPBroadCast):
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(outType, ndsbp, a.Placement);
    }

    public static IRType VisitTensorType(TensorType lhs, TensorType rhs, bool packingK = false, DimInfo? dimInfo = null)
    {
        if (lhs.Shape.IsUnranked || rhs.Shape.IsUnranked)
        {
            return new TensorType(lhs.DType, Shape.Unranked);
        }

        var (lm, lk, rk, rn) = dimInfo ?? new(lhs.Shape.Rank - 2, lhs.Shape.Rank - 1, rhs.Shape.Rank - 2, rhs.Shape.Rank - 1);
        DataType dtype = lhs.DType;
        DataType lhsDType = lhs.DType is VectorType l ? l.ElemType : lhs.DType;
        DataType rhsDType = rhs.DType is VectorType r ? r.ElemType : rhs.DType;
        if (lhsDType != rhsDType)
        {
            return new InvalidType("MatMul lhs and rhs have different DType");
        }

        if (lhs.Shape[lk] != rhs.Shape[rk] && lhs.Shape[lk] != Dimension.Unknown && rhs.Shape[rk] != Dimension.Unknown)
        {
            return new InvalidType("MatMul lhs and rhs have not compatiable shape");
        }

        if (lhsDType == DataTypes.Float8E4M3 || lhsDType == DataTypes.Float8E5M2 || lhsDType == DataTypes.Int8)
        {
            dtype = DataTypes.Float32;
        }

        if (lhs.DType is VectorType vl1 && rhs.DType is not VectorType)
        {
            if (vl1.Lanes.Count != 1)
            {
                return new InvalidType("Only packing m is supported when rhs is not vector type.");
            }

            dtype = vl1;
        }
        else if (lhs.DType is not VectorType && rhs.DType is VectorType vr1)
        {
            if (vr1.Lanes.Count != 1)
            {
                return new InvalidType("Only packing n is supported when lhs is not vector type.");
            }

            dtype = vr1;
        }
        else if (lhs.DType is VectorType vl && rhs.DType is VectorType vr)
        {
            // pack k or m&n
            var elemType = vl.ElemType;
            if (elemType == DataTypes.Float8E4M3 || elemType == DataTypes.Float8E5M2)
            {
                elemType = DataTypes.Float32;
            }

            if (vl.Lanes.Count == 1 && vr.Lanes.Count == 1)
            {
                dtype = packingK ? elemType : new VectorType(elemType, vl.Lanes[0], vr.Lanes[0]);
            }
            else if (vl.Lanes.Count == 1 && vr.Lanes.Count == 2)
            {
                dtype = new VectorType(elemType, vr.Lanes[1]);
            }
            else if (vl.Lanes.Count == 2 && vr.Lanes.Count == 1)
            {
                dtype = new VectorType(elemType, vl.Lanes[0]);
            }
            else if (vl.Lanes.Count == 2 && vr.Lanes.Count == 2)
            {
                // TODO: only support transpose vector B for now
                dtype = new VectorType(elemType, vl.Lanes[0], vl.Lanes[1] == vr.Lanes[0] ? vr.Lanes[1] : vr.Lanes[0]);
            }
            else
            {
                return new InvalidType("Not supported packing.");
            }
        }

        var lhsShape = lhs.Shape.Rank >= rhs.Shape.Rank ? lhs.Shape.ToArray() : Enumerable.Repeat((Dimension)1, rhs.Shape.Rank - lhs.Shape.Rank).Concat(lhs.Shape).ToArray();
        var rhsShape = lhs.Shape.Rank <= rhs.Shape.Rank ? rhs.Shape.ToArray() : Enumerable.Repeat((Dimension)1, lhs.Shape.Rank - rhs.Shape.Rank).Concat(rhs.Shape).ToArray();

        var bigShape = Enumerable.Zip(lhsShape, rhsShape).SkipLast(2).Select(t =>
            t.First == Dimension.Unknown || t.Second == Dimension.Unknown
                ? Dimension.Unknown
                : System.Math.Max(t.First.FixedValue, t.Second.FixedValue)).ToArray();

        // batch and channel
        var front = bigShape;

        // currently the output keep the m,n.
        var end = new[] { lhs.Shape[lm], rhs.Shape[rn] };
        return new TensorType(dtype, front.Concat(end).ToArray());
    }

    public static IValue InferValue(DataType dataType, Tensor lhs, Tensor rhs)
    {
        if (dataType == DataTypes.Float8E4M3 || dataType == DataTypes.Float8E5M2)
        {
            var lhsOrt = Cast(lhs, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
            var rhsOrt = Cast(rhs, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
            var ret = OrtKI.MatMul(lhsOrt, rhsOrt).ToTensor();
            return Value.FromTensor(ret);
        }
        else
        {
            var input = lhs.ToOrtTensor();
            var other = rhs.ToOrtTensor();
            return OrtKI.MatMul(input, other).ToValue();
        }
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, MatMul matMul)
    {
        var dataType = context.CurrentCall.Arguments[MatMul.Lhs.Index].CheckedDataType;
        var lhs = context.GetArgumentValue(matMul, MatMul.Lhs).AsTensor();
        var rhs = context.GetArgumentValue(matMul, MatMul.Rhs).AsTensor();
        return InferValue(dataType, lhs, rhs);
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
            _ => new InvalidType($"{lhs} {rhs} not support"),
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
            var lhsType = DistributedUtility.GetDividedTensorType(distributedType);
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

    public record DimInfo(int Lm, int Lk, int Rk, int Rn)
    {
    }
}
