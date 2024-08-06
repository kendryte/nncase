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
/// Evaluator for <see cref="NcnnSoftplus"/>.
/// </summary>
public class NcnnSoftplusEvaluator : IEvaluator<NcnnSoftplus>, ITypeInferencer<NcnnSoftplus>, ICostEvaluator<NcnnSoftplus>, IShapeEvaluator<NcnnSoftplus>, IMetricEvaluator<NcnnSoftplus>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnSoftplus softplus)
    {
        var input = context.GetOrtArgumentValue(softplus, NcnnSoftplus.Input);
        return OrtKI.Softplus(input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnSoftplus target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnSoftplus.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnSoftplus target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnSoftplus target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnSoftplus.Input);

        return new()
        {
            // y = log(exp(x)+1).
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.AddFLOPs + MetricUtility.ExpFLOPs + MetricUtility.LogFLOPs),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnSoftplus target) => context.GetArgumentShape(target, NcnnSoftplus.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
