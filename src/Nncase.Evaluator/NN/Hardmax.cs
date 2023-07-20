// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Hardmax"/>.
/// </summary>
public class HardmaxEvaluator : IEvaluator<Hardmax>, ITypeInferencer<Hardmax>, ICostEvaluator<Hardmax>, IMetricEvaluator<Hardmax>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Hardmax target)
    {
        var input = context.GetOrtArgumentValue(target, Hardmax.Input);
        var axis = context.GetArgumentValueAsScalar<long>(target, Hardmax.Axis);
        return OrtKI.Hardmax(input, axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Hardmax target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Hardmax.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Hardmax target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Hardmax.Input);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = (UInt128)inputType.Shape.Rank,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Hardmax target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Hardmax.Input);
        var returnType = context.GetReturnType<TensorType>();
        var returnF = MetricUtility.GetFLOPs(returnType);
        var inputF = MetricUtility.GetFLOPs(inputType);
        var inner = inputF / returnF;
        var outter = inputF / inner;

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = outter * inner,
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
