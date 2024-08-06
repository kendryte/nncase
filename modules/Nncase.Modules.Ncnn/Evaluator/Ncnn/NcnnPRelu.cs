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
/// Evaluator for <see cref="NcnnPReLU"/>.
/// </summary>
public class NcnnPReluEvaluator : IEvaluator<NcnnPReLU>, ITypeInferencer<NcnnPReLU>, ICostEvaluator<NcnnPReLU>, IShapeEvaluator<NcnnPReLU>, IMetricEvaluator<NcnnPReLU>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnPReLU pReLU)
    {
        var input = context.GetOrtArgumentValue(pReLU, NcnnPReLU.Input);
        var slope = pReLU.Slope;
        return OrtKI.PRelu(input, slope).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnPReLU target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnPReLU.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnPReLU target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnPReLU target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnPReLU.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.CmpFLOPs + MetricUtility.MulFLOPs),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnPReLU target) => context.GetArgumentShape(target, NcnnPReLU.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
