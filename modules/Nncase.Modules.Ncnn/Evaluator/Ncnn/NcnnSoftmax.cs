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
/// Evaluator for <see cref="NcnnSoftmax"/>.
/// </summary>
public class NcnnSoftmaxEvaluator : IEvaluator<NcnnSoftmax>, ITypeInferencer<NcnnSoftmax>, ICostEvaluator<NcnnSoftmax>, IShapeEvaluator<NcnnSoftmax>, IMetricEvaluator<NcnnSoftmax>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnSoftmax softmax)
    {
        var input = context.GetOrtArgumentValue(softmax, NcnnSoftmax.Input);
        var dim = softmax.Axis;
        return OrtKI.Softmax(input, dim).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnSoftmax target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnSoftmax.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnSoftmax target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnSoftmax target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnSoftmax.Input);
        var returnType = context.GetReturnType<TensorType>();
        var returnF = MetricUtility.GetFLOPs(returnType);
        var inputF = MetricUtility.GetFLOPs(inputType);
        var inner = inputF / returnF;

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = (inner * 2) + (inputF * (MetricUtility.SubFLOPs + MetricUtility.ExpFLOPs + MetricUtility.DivFLOPs)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnSoftmax target) => context.GetArgumentShape(target, NcnnSoftmax.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
