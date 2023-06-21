// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using DryIoc.ImTools;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Rank"/>.
/// </summary>
public class RankEvaluator : IEvaluator<Rank>, ITypeInferencer<Rank>, ICostEvaluator<Rank>, IShapeEvaluator<Rank>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Rank rank)
    {
        var input = context.GetArgumentValue(rank, Rank.Input);
        return Value.FromTensor(input.AsTensor().Shape.Rank);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Rank target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Rank.Input);
        return Visit(context, target, input);
    }

    public Cost Visit(ICostEvaluateContext context, Rank target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Rank.Input);
        var outputType = context.GetReturnType<TensorType>();
        return new Cost()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, CostUtility.GetCPUCyclesOfBinary(BinaryOp.Mul)),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Rank target) => 1;

    private IRType Visit(ITypeInferenceContext context, Rank target, TensorType input)
    {
        return new TensorType(input.DType, Shape.Scalar);
    }
}
