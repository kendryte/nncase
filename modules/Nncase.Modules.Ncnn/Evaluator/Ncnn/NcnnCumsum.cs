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
/// Evaluator for <see cref="NcnnCumsum"/>.
/// </summary>
public class NcnnCumsumEvaluator : IEvaluator<NcnnCumsum>, ITypeInferencer<NcnnCumsum>, ICostEvaluator<NcnnCumsum>, IShapeEvaluator<NcnnCumsum>, IMetricEvaluator<NcnnCumsum>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnCumsum cumsum)
    {
        var input = context.GetOrtArgumentValue(cumsum, NcnnCumsum.Input);
        var axis = cumsum.Axis;
        var axisTensor = new Tensor<int>(new[] { axis }, new Shape(1));
        return OrtKI.CumSum(input, axisTensor.ToOrtTensor(), 0L, 0L).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnCumsum target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnCumsum.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnCumsum target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnCumsum target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnCumsum.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * MetricUtility.AddFLOPs,
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnCumsum target) => context.GetArgumentShape(target, NcnnCumsum.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
