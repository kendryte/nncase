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
/// Evaluator for <see cref="NcnnLRN"/>.
/// </summary>
public class NcnnLRNEvaluator : IEvaluator<NcnnLRN>, ITypeInferencer<NcnnLRN>, ICostEvaluator<NcnnLRN>, IShapeEvaluator<NcnnLRN>, IMetricEvaluator<NcnnLRN>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnLRN lrn)
    {
        var input = context.GetOrtArgumentValue(lrn, NcnnLRN.Input);
        var alpha = lrn.Alpha;
        var beta = lrn.Beta;
        var bias = lrn.Bias;
        var size = lrn.Size;
        return OrtKI.LRN(input, alpha, beta, bias, size).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnLRN target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnLRN.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnLRN target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnLRN target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnLRN.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.ExpFLOPs + MetricUtility.MulFLOPs + MetricUtility.SubFLOPs + MetricUtility.CmpFLOPs),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnLRN target) => context.GetArgumentShape(target, NcnnLRN.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
