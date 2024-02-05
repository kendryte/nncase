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
/// Evaluator for <see cref="NcnnPRelu"/>.
/// </summary>
public class NcnnPReluEvaluator : IEvaluator<NcnnPRelu>, ITypeInferencer<NcnnPRelu>, ICostEvaluator<NcnnPRelu>, IShapeEvaluator<NcnnPRelu>, IMetricEvaluator<NcnnPRelu>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnPRelu pRelu)
    {
        var input = context.GetOrtArgumentValue(pRelu, NcnnPRelu.Input);
        var slope = pRelu.Slope;
        return OrtKI.PRelu(input, slope).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnPRelu target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnPRelu.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnPRelu target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnPRelu target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnPRelu.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.CmpFLOPs + MetricUtility.MulFLOPs),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnPRelu target) => context.GetArgumentShape(target, NcnnPRelu.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
