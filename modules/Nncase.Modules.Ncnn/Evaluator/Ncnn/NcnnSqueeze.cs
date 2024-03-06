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
/// Evaluator for <see cref="NcnnSqueeze"/>.
/// </summary>
public class NcnnSqueezeEvaluator : IEvaluator<NcnnSqueeze>, ITypeInferencer<NcnnSqueeze>, ICostEvaluator<NcnnSqueeze>, IShapeEvaluator<NcnnSqueeze>, IMetricEvaluator<NcnnSqueeze>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnSqueeze squeeze)
    {
        var input = context.GetOrtArgumentValue(squeeze, NcnnSqueeze.Input);
        var dims = squeeze.Dims;
        return OrtKI.Squeeze(input, dims).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnSqueeze target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnSqueeze.Input);
        var dims = target.Dims;
        return Visit(input, dims);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnSqueeze target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnSqueeze target)
    {
        _ = context.GetArgumentType<TensorType>(target, NcnnSqueeze.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = 0,
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnSqueeze target) => context.GetArgumentShape(target, NcnnSqueeze.Input);

    private IRType Visit(TensorType input, int[] dims)
    {
        var outputShape = input.Shape.ToValueArray();
        if (dims.Length == 0)
        {
            outputShape = outputShape.Where(x => x != 1).ToArray();
        }
        else
        {
            // outputShape = outputShape.Select((value, idx) => dims.Contains(idx) ? value : (value == 1 ? 0 : value)).Where(v => v != 0).ToArray();
            outputShape = outputShape.Select((value, idx) => dims.Contains(idx) ? (value == 1 ? 0 : value) : value).Where(v => v != 0)
                .ToArray();
        }


        return new TensorType(input.DType, outputShape);
    }
}
