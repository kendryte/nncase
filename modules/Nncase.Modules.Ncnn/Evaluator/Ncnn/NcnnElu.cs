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
/// Evaluator for <see cref="NcnnElu"/>.
/// </summary>
public class NcnnEluEvaluator : IEvaluator<NcnnElu>, ITypeInferencer<NcnnElu>, ICostEvaluator<NcnnElu>, IShapeEvaluator<NcnnElu>, IMetricEvaluator<NcnnElu>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnElu elu)
    {
        var input = context.GetOrtArgumentValue(elu, NcnnElu.Input);
        var alpha = elu.Alpha;
        return OrtKI.Elu(input, alpha).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnElu target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnElu.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnElu target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnElu target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnElu.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.ExpFLOPs + MetricUtility.MulFLOPs + MetricUtility.SubFLOPs + MetricUtility.CmpFLOPs),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnElu target) => context.GetArgumentShape(target, NcnnElu.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
