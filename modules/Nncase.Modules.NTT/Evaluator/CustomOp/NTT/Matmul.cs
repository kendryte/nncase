// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.TIR.NTT;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using MatMul = Nncase.IR.CustomNTT.MatMul;

namespace Nncase.Evaluator.CustomNTT;

/// <summary>
/// Evaluator for <see cref="MatMul"/>.
/// </summary>
public class MatMulEvaluator : IEvaluator<MatMul>, ITypeInferencer<MatMul>, ICostEvaluator<MatMul>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, MatMul matMul)
    {
        // TODO: rewrite this to use OrtKISharp
        var dataType = context.CurrentCall.Arguments[Matmul.Lhs.Index].CheckedDataType;
        var lhs = context.GetArgumentValue(matMul, Matmul.Lhs).AsTensor();
        var rhs = context.GetArgumentValue(matMul, MatMul.Rhs).AsTensor();
        return Math.MatMulEvaluator.InferValue(dataType, lhs, rhs);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, MatMul target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, MatMul.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, MatMul.Rhs);
        var (lhsRank, rhsRank) = (lhs, rhs) switch
        {
            (DistributedType a, DistributedType b) => (a.TensorType.Shape.Rank, b.TensorType.Shape.Rank),
            (TensorType a, TensorType b) => (a.Shape.Rank, b.Shape.Rank),
            _ => throw new ArgumentException($"Unsupported type: {lhs} {rhs}"),
        };
        var dimInfo = new Math.MatMulEvaluator.DimInfo(target.TransposeA ? lhsRank - 1 : lhsRank - 2, target.TransposeA ? lhsRank - 2 : lhsRank - 1, target.TransposeB ? rhsRank - 1 : rhsRank - 2, target.TransposeB ? rhsRank - 2 : rhsRank - 1);

        if (CheckCustomSBP(lhs, rhs, target))
        {
            return (lhs, rhs) switch
            {
                (DistributedType a, DistributedType b) => new DistributedType((TensorType)VisitTensorType(a.TensorType, b.TensorType, true, dimInfo), target.OutSBPs, a.Placement),
                (TensorType a, TensorType b) => VisitTensorType(a, b, true, dimInfo),
                _ => new InvalidType($"{lhs} {rhs} not support"),
            };
        }
        else
        {
            return new InvalidType("Not Match With CustomSBP!");
        }
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, MatMul target)
    {
        return target.Cost;
    }

    private bool CheckCustomSBP(IRType lhs, IRType rhs, MatMul matmul)
    {
        if (lhs is DistributedType a && rhs is DistributedType b)
        {
            if (Enumerable.Range(0, a.TensorType.Shape.Rank).Any(i => a.AxisPolicies[i] != matmul.LhsSBPs[i]))
            {
                return false;
            }

            if (Enumerable.Range(0, b.TensorType.Shape.Rank).Any(i => b.AxisPolicies[i] != matmul.RhsSBPs[i]))
            {
                return false;
            }
        }

        return true;
    }

    private IRType VisitTensorType(TensorType lhs, TensorType rhs, bool packingK = false, Math.MatMulEvaluator.DimInfo? dimInfo = null)
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

        if (lhs.Shape[lk] != rhs.Shape[rk] && lhs.Shape[lk].IsFixed && rhs.Shape[rk].IsFixed)
        {
            return new InvalidType("MatMul lhs and rhs have not compatiable shape");
        }

        if (lhsDType == DataTypes.Float8E4M3 || lhsDType == DataTypes.Float8E5M2 || lhsDType == DataTypes.Int8)
        {
            dtype = DataTypes.Float32;
        }

        if (lhs.DType is VectorType vl && rhs.DType is VectorType vr)
        {
            // pack k or m&n
            var lhsElemType = vl.ElemType;
            var outElementType = lhsElemType;
            if (lhsElemType.IsFloat() && lhsElemType != DataTypes.Float32)
            {
                outElementType = DataTypes.Float32;
            }

            // TODO: support other custom packing
            if (vl.Lanes.Count == 1 && vr.Lanes.Count == 1)
            {
                var scale = 1f * outElementType.SizeInBytes / lhsElemType.SizeInBytes;
                dtype = packingK ? new VectorType(outElementType, (int)(vl.Lanes[0] / scale)) : new VectorType(outElementType, vl.Lanes[0], vr.Lanes[0]);
            }
            else
            {
                return new InvalidType("Not supported packing.");
            }
        }

        var lhsShape = lhs.Shape.Rank >= rhs.Shape.Rank ? lhs.Shape.ToArray() : Enumerable.Repeat((Dimension)1, rhs.Shape.Rank - lhs.Shape.Rank).Concat(lhs.Shape).ToArray();
        var rhsShape = lhs.Shape.Rank <= rhs.Shape.Rank ? rhs.Shape.ToArray() : Enumerable.Repeat((Dimension)1, lhs.Shape.Rank - rhs.Shape.Rank).Concat(rhs.Shape).ToArray();

        var bigShape = Enumerable.Zip(lhsShape, rhsShape).SkipLast(2).Select(t => Dimension.Max(t.First, t.Second)).ToArray();

        // batch and channel
        var front = bigShape;

        var end = new[] { lhs.Shape[lm], rhs.Shape[rn] };
        if (lhs.DType is VectorType && rhs.DType is VectorType)
        {
            end = new[] { rhs.Shape[rn] / ((VectorType)dtype).Lanes[0], lhs.Shape[lm] };
        }

        return new TensorType(dtype, front.Concat(end).ToArray());
    }
}
