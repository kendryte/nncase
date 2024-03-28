// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnUnsqueeze"/>.
/// </summary>
public class NcnnUnsqueezeEvaluator : IEvaluator<NcnnUnsqueeze>, ITypeInferencer<NcnnUnsqueeze>, ICostEvaluator<NcnnUnsqueeze>, IShapeEvaluator<NcnnUnsqueeze>, IMetricEvaluator<NcnnUnsqueeze>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnUnsqueeze unsqueeze)
    {
        var input = context.GetOrtArgumentValue(unsqueeze, NcnnUnsqueeze.Input);
        var dims = unsqueeze.Dims;
        return OrtKI.Unsqueeze(input, dims).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnUnsqueeze target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnUnsqueeze.Input);
        var dims = target.Dims;
        return Visit(input, dims);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnUnsqueeze target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnUnsqueeze target)
    {
        _ = context.GetArgumentType<TensorType>(target, NcnnUnsqueeze.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = 0,
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnUnsqueeze target) => context.GetArgumentShape(target, NcnnUnsqueeze.Input);

    private IRType Visit(TensorType input, int[] dims)
    {
        var outputShape = input.Shape.ToValueArray().ToList();
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            outputShape.Insert(i, 1);
        }

        return new TensorType(input.DType, outputShape.ToArray());
    }
}
