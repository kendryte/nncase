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
/// Evaluator for <see cref="NcnnReshape"/>.
/// </summary>
public class NcnnReshapeEvaluator : IEvaluator<NcnnReshape>, ITypeInferencer<NcnnReshape>, ICostEvaluator<NcnnReshape>, IShapeEvaluator<NcnnReshape>, IMetricEvaluator<NcnnReshape>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnReshape reshape)
    {
        var input = context.GetOrtArgumentValue(reshape, NcnnReshape.Input);
        var newShape = reshape.Shape;
        return OrtKI.Reshape(input, newShape, 0).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnReshape target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnReshape.Input);
        var newShape = target.Shape;
        return Visit(input, newShape);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnReshape target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnReshape target)
    {
        _ = context.GetArgumentType<TensorType>(target, NcnnReshape.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = 0,
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnReshape target) => context.GetArgumentShape(target, NcnnReshape.Input);

    private IRType Visit(TensorType input, int[] newShape)
    {
        var inputSize = input.Shape.Aggregate((a, b) => a * b).FixedValue;
        var outputShape = newShape;
        int negAxis = -1;
        for (int i = 0; i < newShape.Length; i++)
        {
            if (newShape[i] == -1)
            {
                negAxis = i;
            }
        }

        if (negAxis != -1)
        {
            int otherSize = newShape.Aggregate(1, (currentProduct, num) => num == 0 ? currentProduct : currentProduct * (num > 0 ? num : 1));
            outputShape[negAxis] = inputSize / otherSize;
        }

        return new TensorType(input.DType, outputShape);
    }
}
