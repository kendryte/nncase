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
/// Evaluator for <see cref="NcnnErf"/>.
/// </summary>
public class NcnnErfEvaluator : IEvaluator<NcnnErf>, ITypeInferencer<NcnnErf>, ICostEvaluator<NcnnErf>, IShapeEvaluator<NcnnErf>, IMetricEvaluator<NcnnErf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnErf erf)
    {
        var input = context.GetOrtArgumentValue(erf, NcnnErf.Input);
        return OrtKI.Erf(input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnErf target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnErf.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnErf target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnErf target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnErf.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.ExpFLOPs + MetricUtility.MulFLOPs + MetricUtility.SubFLOPs + MetricUtility.CmpFLOPs ),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnErf target) => context.GetArgumentShape(target, NcnnErf.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
