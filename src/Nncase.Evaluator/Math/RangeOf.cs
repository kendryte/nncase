// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Numerics;
using System.Runtime.Intrinsics.X86;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="RangeOf"/>.
/// </summary>
public class RangeOfEvaluator : IEvaluator<RangeOf>, ITypeInferencer<RangeOf>, ICostEvaluator<RangeOf>, IShapeEvaluator<RangeOf>, IMetricEvaluator<RangeOf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, RangeOf target)
    {
        var input = context.GetArgumentValueAsTensor<float>(target, RangeOf.Input);
        var min = float.MaxValue;
        var max = float.MinValue;
        foreach (var f in input.Buffer.Span)
        {
            if (float.IsFinite(f))
            {
                min = System.Math.Min(min, f);
                max = System.Math.Max(max, f);
            }
        }

        return Value.FromTensor(new[] { min, max });
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, RangeOf target)
    {
        var input = context.CheckArgumentType<TensorType>(target, RangeOf.Input);
        return input with { Shape = new Shape(2) };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, RangeOf target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, RangeOf.Input);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(inputType, 2),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, RangeOf target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, RangeOf.Input);
        _ = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType, 2),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, RangeOf target) => context.GetArgumentShape(target, RangeOf.Input);
}
