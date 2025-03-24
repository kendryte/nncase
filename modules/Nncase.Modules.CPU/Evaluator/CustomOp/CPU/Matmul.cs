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

        if (CheckCustomSBP(lhs, rhs, target))
        {
            return (lhs, rhs) switch
            {
                (DistributedType a, DistributedType b) => Math.MatMulEvaluator.VisitDistributedType(a, b),
                (TensorType a, TensorType b) => Math.MatMulEvaluator.VisitTensorType(a, b),
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
            for (int i = 0; i < a.Placement.Rank; i++)
            {
                if (a.AxisPolices[i] != matmul.LhsSBPs[i])
                {
                    return false;
                }

                if (b.AxisPolices[i] != matmul.RhsSBPs[i])
                {
                    return false;
                }
            }
        }

        return true;
    }
}
