// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="CumSum"/>.
/// </summary>
public class CumSumEvaluator : IEvaluator<CumSum>, ITypeInferencer<CumSum>, ICostEvaluator<CumSum>, IShapeEvaluator<CumSum>, IMetricEvaluator<CumSum>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, CumSum cumSum)
    {
        var input = context.GetOrtArgumentValue(cumSum, CumSum.Input);

        // in onnx, CumSum.Axis is a input tensor with one value
        var axis = context.GetOrtArgumentValue(cumSum, CumSum.Axis);
        var exclusive = context.GetArgumentValueAsScalar<long>(cumSum, CumSum.Exclusive);
        var reverse = context.GetArgumentValueAsScalar<long>(cumSum, CumSum.Reverse);
        return OrtKI.CumSum(input, axis, exclusive, reverse).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, CumSum target)
    {
        var input = context.CheckArgumentType<TensorType>(target, CumSum.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, CumSum target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, CumSum.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, CumSum target) => context.GetArgumentShape(target, CumSum.Input);

    public Metric Visit(IMetricEvaluateContext context, CumSum target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, CumSum.Input);
        var returnType = context.GetReturnType<TensorType>();
        var returnF = MetricUtility.GetFLOPs(returnType);
        var inputF = MetricUtility.GetFLOPs(inputType);
        var inner = inputF / returnF;
        _ = inputF / inner;
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = inner * 2,
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
