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
/// Evaluator for <see cref="NcnnSigmoid"/>.
/// </summary>
public class NcnnSigmoidEvaluator : IEvaluator<NcnnSigmoid>, ITypeInferencer<NcnnSigmoid>, ICostEvaluator<NcnnSigmoid>, IShapeEvaluator<NcnnSigmoid>, IMetricEvaluator<NcnnSigmoid>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnSigmoid sigmoid)
    {
        var input = context.GetOrtArgumentValue(sigmoid, NcnnSigmoid.Input);
        return OrtKI.Sigmoid(input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnSigmoid target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnSigmoid.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnSigmoid target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnSigmoid target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnSigmoid.Input);

        return new()
        {
            // y = 1 / (1 + exp(-x)).
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.AddFLOPs + MetricUtility.ExpFLOPs + MetricUtility.DivFLOPs + MetricUtility.MulFLOPs),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnSigmoid target) => context.GetArgumentShape(target, NcnnSigmoid.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
