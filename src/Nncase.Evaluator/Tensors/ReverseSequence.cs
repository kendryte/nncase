// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="ReverseSequence"/>.
/// </summary>
public class ReverseSequenceEvaluator : IEvaluator<ReverseSequence>, ITypeInferencer<ReverseSequence>, ICostEvaluator<ReverseSequence>, IMetricEvaluator<ReverseSequence>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ReverseSequence random)
    {
        var input = context.GetOrtArgumentValue(random, ReverseSequence.Input);
        var seqLens = context.GetOrtArgumentValue(random, ReverseSequence.SeqLens);
        var batchAxis = context.GetArgumentValueAsScalar<long>(random, ReverseSequence.BatchAxis);
        var timeAxis = context.GetArgumentValueAsScalar<long>(random, ReverseSequence.TimeAxis);
        return OrtKI.ReverseSequence(input, seqLens, batchAxis, timeAxis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ReverseSequence target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ReverseSequence.Input);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, ReverseSequence target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, ReverseSequence.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, ReverseSequence target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
        };
    }

    private IRType Visit(ITypeInferenceContext context, ReverseSequence target, TensorType input)
    {
        return input;
    }
}
