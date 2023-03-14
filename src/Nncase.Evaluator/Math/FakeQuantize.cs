// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="FakeQuantize"/>.
/// </summary>
public class FakeQuantizeEvaluator : IEvaluator<FakeQuantize>, ITypeInferencer<FakeQuantize>, ICostEvaluator<FakeQuantize>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, FakeQuantize target)
    {
        var input = context.GetArgumentValue(target, FakeQuantize.Input);
        return input;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, FakeQuantize target)
    {
        var input = context.CheckArgumentType<TensorType>(target, FakeQuantize.Input);
        var quantParam = context.CheckArgumentType<TensorType>(target, FakeQuantize.QuantParam);
        return Visit(target, input, quantParam);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, FakeQuantize target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, FakeQuantize.Input);
        var outputType = context.GetReturnType<TensorType>();

        // TODO: bits
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetFakeMemoryAccess(outputType, 8),
            [CostFactorNames.CPUCycles] = 2,
        };
    }

    private IRType Visit(FakeQuantize target, TensorType input, TensorType quantParam)
    {
        return input;
    }
}
