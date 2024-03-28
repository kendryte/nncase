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
/// Evaluator for <see cref="NcnnSELU"/>.
/// </summary>
public class NcnnSELUEvaluator : IEvaluator<NcnnSELU>, ITypeInferencer<NcnnSELU>, ICostEvaluator<NcnnSELU>, IShapeEvaluator<NcnnSELU>, IMetricEvaluator<NcnnSELU>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnSELU selu)
    {
        var input = context.GetOrtArgumentValue(selu, NcnnSELU.Input);
        var alpha = selu.Alpha;
        var gamma = selu.Gamma;
        return OrtKI.Selu(input, alpha, gamma).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnSELU target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnSELU.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnSELU target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnSELU target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnSELU.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] =
                MetricUtility.GetFLOPs(inputType) * (MetricUtility.ExpFLOPs + (MetricUtility.MulFLOPs * 2) +
                                                     MetricUtility.SubFLOPs + MetricUtility.AddFLOPs +
                                                     (MetricUtility.CmpFLOPs * 2)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnSELU target) => context.GetArgumentShape(target, NcnnSELU.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
