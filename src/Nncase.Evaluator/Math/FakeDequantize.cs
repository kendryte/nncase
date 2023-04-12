// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="FakeDequantize"/>.
/// </summary>
public class FakeDequantizeEvaluator : IEvaluator<FakeDequantize>, ITypeInferencer<FakeDequantize>, ICostEvaluator<FakeDequantize>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, FakeDequantize target)
    {
        var input = context.GetArgumentValue(target, FakeDequantize.Input);
        return input;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, FakeDequantize target)
    {
        var input = context.CheckArgumentType<TensorType>(target, FakeDequantize.Input);
        var quantParam = context.CheckArgumentType<TensorType>(target, FakeDequantize.DequantParam);
        return Visit(target, input, quantParam);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, FakeDequantize target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, FakeDequantize.Input);
        var outputType = context.GetReturnType<TensorType>();

        // TODO: bits
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetFakeMemoryAccess(inputType, 8),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = 2,
        };
    }

    private IRType Visit(FakeDequantize target, TensorType input, TensorType quantParam)
    {
        return input;
    }
}
