// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnClip"/>.
/// </summary>
public class NcnnClipEvaluator : IEvaluator<NcnnClip>, ITypeInferencer<NcnnClip>, ICostEvaluator<NcnnClip>, IShapeEvaluator<NcnnClip>, IMetricEvaluator<NcnnClip>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnClip clip)
    {
        var input = context.GetOrtArgumentValue(clip, NcnnClip.Input);
        var min = clip.Min;
        var max = clip.Max;
        return OrtKI.Clip(input, min, max).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnClip target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnClip.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnClip target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnClip target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnClip.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * 2,
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnClip target) => context.GetArgumentShape(target, NcnnClip.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
