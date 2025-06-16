// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Math;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using MatMul = Nncase.IR.Math.MatMul;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="MatMul"/>.
/// </summary>
public class MatMulEvaluator : IEvaluator<MatMul>, ITypeInferencer<MatMul>, ICostEvaluator<MatMul>, IMetricEvaluator<MatMul>
{
    public static IRType VisitDistributedType(DistributedType a, DistributedType b, bool packingK = false, DimInfo? dimInfo = null, bool transB = false, DataType outputDataType = null!)
    {
        if (VisitTensorType(a.TensorType, b.TensorType, packingK, dimInfo, outputDataType) is not TensorType outType)
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
        var (lm, lk, rk, rn) = dimInfo ?? new(aRank - 2, aRank - 1, bRank - 2, bRank - 1);
        var aMaxShape = CompilerServices.GetMaxShape(a.TensorType.Shape);
        var bMaxShape = CompilerServices.GetMaxShape(b.TensorType.Shape);

        // TODO: keep summa only
        if (!a.TensorType.Shape.IsFixed || !b.TensorType.Shape.IsFixed || transB || (a.Placement.HierarchyKind == HierarchyKind.SMT && a.TensorType.DType is VectorType vt && vt.ElemType == DataTypes.Float8E4M3))
        {
            var ndsbp = new SBP[oRank];
            for (int i = 0; i < ndsbp.Length - 2; i++)
            {
                var policyA = i < aPad ? null : a.AxisPolicies[i - aPad];
                var policyB = i < bPad ? null : b.AxisPolicies[i - bPad];
                var invalid = new InvalidType($"({policyA}, {policyB}) not support");
                switch (policyA, policyB)
                {
                    case (null, _):
                        ndsbp[i] = policyB!;
                        break;
                    case (_, null):
                        ndsbp[i] = policyA!;
                        break;
                    case (SBPSplit sa, SBPSplit sb):
                        if (sa.Axes != sb.Axes)
                        {
                            return invalid;
                        }

                        ndsbp[i] = sa;
                        break;
                    case (SBPSplit sa, SBPBroadCast):
                        // invalid (S, B) if B is not broacast
                        if (i < lm + aPad && bMaxShape[i - bPad] != 1)
                        {
                            return invalid;
                        }

                        ndsbp[i] = sa;
                        break;
                    case (SBPBroadCast, SBPSplit sb):
                        // invalid (B, S) if A is not broacast
                        if (i < rk + bPad && aMaxShape[i - aPad] != 1)
                        {
                            return invalid;
                        }

                        ndsbp[i] = sb;
                        break;
                    case (SBPBroadCast, SBPBroadCast):
                        ndsbp[i] = SBP.B;
                        break;
                    default:
                        return invalid;
                }
            }

            if (a.AxisPolicies[lk] is SBPSplit || b.AxisPolicies[rk] is SBPSplit)
            {
                return new InvalidType("not support split on k");
            }

            ndsbp[oRank - 2] = a.AxisPolicies[lm];
            ndsbp[oRank - 1] = b.AxisPolicies[rn];

            if (!DistributedUtility.IsDistributable(outType, ndsbp, a.Placement))
            {
                return new InvalidType("no valid sbp.");
            }

            return new DistributedType(outType, ndsbp, a.Placement);
        }
        else
        {
            var ndsbp = new SBP[oRank];
            if (a.Placement.Rank == 1)
            {
                // not support split on k.
                if (a.AxisPolicies[lk] is SBPSplit || b.AxisPolicies[rk] is SBPSplit)
                {
                    return new InvalidType("not support split on k for 1d mesh.");
                }

                ndsbp[oRank - 2] = a.AxisPolicies[lm];
                ndsbp[oRank - 1] = b.AxisPolicies[rn];
            }
            else
            {
                if (a.AxisPolicies[lk] is SBPSplit || b.AxisPolicies[rk] is SBPSplit)
                {
                    var (lmMeshAxis, lkMeshAxis) = (a.Placement.Rank - 2, a.Placement.Rank - 1);

                    // TODO: support split on multi-meshes.
                    if (a.AxisPolicies[lm] is SBPSplit slm && a.AxisPolicies[lk] is SBPSplit slk
                    && b.AxisPolicies[rk] is SBPSplit srk && b.AxisPolicies[rn] is SBPSplit srn
                    && slm.Axes.Count == 1 && slk.Axes.Count == 1 && srk.Axes.Count == 1 && srn.Axes.Count == 1
                    && slm.Axes[0] == srk.Axes[0] && slk.Axes[0] == srn.Axes[0]
                    && slm.Axes[0] == lmMeshAxis && slk.Axes[0] == lkMeshAxis)
                    {
                        ndsbp[oRank - 2] = a.AxisPolicies[lm];
                        ndsbp[oRank - 1] = b.AxisPolicies[rn];
                    }
                    else
                    {
                        return new InvalidType("only support specific split for summa.");
                    }
                }
                else
                {
                    ndsbp[oRank - 2] = a.AxisPolicies[lm];
                    ndsbp[oRank - 1] = b.AxisPolicies[rn];
                }
            }

            for (int i = 0; i < ndsbp.Length - 2; i++)
            {
                var policyA = i < aPad ? null : a.AxisPolicies[i - aPad];
                var policyB = i < bPad ? null : b.AxisPolicies[i - bPad];
                switch (policyA, policyB)
                {
                    case (null, _):
                        ndsbp[i] = policyB!;
                        break;
                    case (_, null):
                        ndsbp[i] = policyA!;
                        break;
                    case (SBPSplit sa, SBPSplit sb):
                        if (sa.Axes != sb.Axes)
                        {
                            return new InvalidType($"lhs rhs sbp at {i} not equal");
                        }

                        ndsbp[i] = sa;
                        break;
                    case (SBPSplit sa, SBPBroadCast):
                        // invalid (S, B) if B is not broacast
                        if (b.TensorType.Shape[i - bPad] != 1)
                        {
                            return new InvalidType($"lhs rhs sbp at {i} not broadcast");
                        }

                        ndsbp[i] = sa;
                        break;
                    case (SBPBroadCast, SBPSplit sb):
                        // invalid (B, S) if A is not broacast
                        if (a.TensorType.Shape[i - aPad] != 1)
                        {
                            return new InvalidType($"lhs rhs sbp at {i} not broadcast");
                        }

                        ndsbp[i] = sb;
                        break;
                    case (SBPBroadCast, SBPBroadCast):
                        ndsbp[i] = SBP.B;
                        break;
                    default:
                        return new InvalidType("not support binary sbp.");
                }
            }

            if (DistributedUtility.IsDistributable(ndsbp))
            {
                return new DistributedType(outType, ndsbp, a.Placement);
            }

            return new InvalidType("no valid sbp.");
        }
    }

    public static IRType ConvertPartialToBroadcast(DistributedType a)
    {
        var ndsbp = a.AxisPolicies.Select(x => x is SBPPartial ? SBP.B : x).ToArray();
        return new DistributedType(a.TensorType, ndsbp, a.Placement);
    }

    public static IRType VisitTensorType(TensorType lhs, TensorType rhs, bool packingK = false, DimInfo? dimInfo = null, DataType outputDataType = null!)
    {
        if (lhs.Shape is not RankedShape lhsShape
            || rhs.Shape is not RankedShape rhsShape)
        {
            return new TensorType(lhs.DType, Shape.Unranked);
        }

        var (lm, lk, rk, rn) = dimInfo ?? new(lhsShape.Rank - 2, lhsShape.Rank - 1, rhsShape.Rank - 2, rhsShape.Rank - 1);
        DataType dtype = lhs.DType;
        DataType lhsDType = lhs.DType is VectorType l ? l.ElemType : lhs.DType;
        DataType rhsDType = rhs.DType is VectorType r ? r.ElemType : rhs.DType;
        if (lhsDType != rhsDType)
        {
            return new InvalidType("MatMul lhs and rhs have different DType");
        }

        if (lhsShape[lk] != rhsShape[rk] && lhsShape[lk].IsFixed && rhsShape[rk].IsFixed)
        {
            return new InvalidType("MatMul lhs and rhs have not compatiable shape");
        }

        if (lhsDType == DataTypes.Float8E4M3 || lhsDType == DataTypes.Float8E5M2 || lhsDType == DataTypes.Int8)
        {
            dtype = outputDataType;
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
            if (vr1.ElemType == DataTypes.Float8E4M3)
            {
                var interType = new VectorType(outputDataType, vr1.Lanes);
                dtype = interType;
            }
        }
        else if (lhs.DType is VectorType vl && rhs.DType is VectorType vr)
        {
            // pack k or m&n
            var elemType = vl.ElemType;
            if (elemType.IsFloat() && elemType != outputDataType)
            {
                elemType = outputDataType;
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
                if (lhsDType == DataTypes.Float16 && vl.Lanes[0] == 64 && vr.Lanes[1] == 64)
                {
                    elemType = lhsDType;
                }

                dtype = new VectorType(elemType, vl.Lanes[0], vl.Lanes[1] == vr.Lanes[0] ? vr.Lanes[1] : vr.Lanes[0]);
            }
            else
            {
                return new InvalidType("Not supported packing.");
            }
        }

        lhsShape = lhsShape.Rank >= rhsShape.Rank ? lhsShape.ToArray() : Enumerable.Repeat((Dimension)1, rhsShape.Rank - lhsShape.Rank).Concat(lhsShape).ToArray();
        rhsShape = lhsShape.Rank <= rhsShape.Rank ? rhsShape.ToArray() : Enumerable.Repeat((Dimension)1, lhsShape.Rank - rhsShape.Rank).Concat(rhsShape).ToArray();

        var bigShape = Enumerable.Zip(lhsShape, rhsShape).SkipLast(2).Select(t => Dimension.Max(t.First, t.Second)).ToArray();

        // batch and channel
        var front = bigShape;

        // currently the output keep the m,n.
        var end = new[] { lhs.Shape[lm], rhs.Shape[rn] };
        return new TensorType(dtype, front.Concat(end).ToArray());
    }

    public static IValue InferValue(DataType dataType, Tensor lhs, Tensor rhs, DataType outputDataType = null!)
    {
        if (dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            var lhsOrt = Cast(lhs, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
            var rhsOrt = Cast(rhs, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
            var result = OrtKI.MatMul(lhsOrt, rhsOrt).ToTensor();
            var ret = Cast(result, outputDataType).Evaluate().AsTensor();
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
        var outputDataType = matMul.OutputDataType;
        return InferValue(dataType, lhs, rhs, outputDataType);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, MatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, MatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, MatMul.Rhs);
        var outputDataType = target.OutputDataType;
        return (lhs, rhs) switch
        {
            (DistributedType a, DistributedType b) => VisitDistributedType(a, b, false, null, false, outputDataType),
            (TensorType a, TensorType b) => VisitTensorType(a, b, false, null, outputDataType),
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
        if (lhs is TensorType { Shape: RankedShape lhsShape })
        {
            macPerElement = lhsShape[^1].IsFixed ? (uint)lhsShape[^1].FixedValue : 1U;
        }
        else if (lhs is DistributedType distributedType)
        {
            var lhsType = DistributedUtility.GetDividedTensorType(distributedType);
            lhsShape = (RankedShape)lhsType.Shape;
            macPerElement = lhsShape[^1].IsFixed ? (uint)lhsShape[^1].FixedValue : 1U;
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
        var lhsShape = (RankedShape)lhs.Shape;
        var k = (UInt128)lhsShape[^1].FixedValue;
        var m = MetricUtility.GetFLOPs(lhs) / k;
        var n = MetricUtility.GetFLOPs(rhs) / k;
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(lhs) + CostUtility.GetMemoryAccess(rhs) + CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = m * n * ((2 * k) - 1),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public record DimInfo(int Lm, int Lk, int Rk, int Rn)
    {
    }
}
