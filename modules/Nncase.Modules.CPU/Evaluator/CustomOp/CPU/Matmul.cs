// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Runtime.CompilerServices;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.TIR.CPU;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using MatMul = Nncase.IR.CustomCPU.MatMul;

namespace Nncase.Evaluator.CustomCPU;

/// <summary>
/// Evaluator for <see cref="MatMul"/>.
/// </summary>
public class MatMulEvaluator : IEvaluator<MatMul>, ITypeInferencer<MatMul>, ICostEvaluator<MatMul>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, MatMul matMul)
    {
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
                (DistributedType a, DistributedType b) => new DistributedType((TensorType)Math.MatMulEvaluator.VisitTensorType(a.TensorType, b.TensorType, true, dimInfo), target.OutSBPs, a.Placement),
                (TensorType a, TensorType b) => Math.MatMulEvaluator.VisitTensorType(a, b, true, dimInfo),
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
            if (Enumerable.Range(0, a.TensorType.Shape.Rank).Any(i => a.AxisPolices[i] != matmul.LhsSBPs[i]))
            {
                return false;
            }

            if (Enumerable.Range(0, b.TensorType.Shape.Rank).Any(i => b.AxisPolices[i] != matmul.RhsSBPs[i]))
            {
                return false;
            }
        }

        return true;
    }
}
