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
using Nncase.IR.Tensors;
using OrtKISharp;
using static Nncase.Evaluator.EvaluatorUtil;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnPooling"/>.
/// </summary>
public class NcnnPoolingEvaluator : IEvaluator<NcnnPooling>, ITypeInferencer<NcnnPooling>, ICostEvaluator<NcnnPooling>, IShapeEvaluator<NcnnPooling>, IMetricEvaluator<NcnnPooling>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnPooling pooling)
    {
        var input = context.GetOrtArgumentValue(pooling, NcnnPooling.Input);
        var kernelSize = new long[] { pooling.Args.KernelH, pooling.Args.KernelW };
        var stride = new long[] { pooling.Args.StrideH, pooling.Args.StrideW };
        var dilation = new long[] { 1, 1 };
        var padData = new long[] { pooling.Args.PadTop, pooling.Args.PadLeft, pooling.Args.PadBottom,  pooling.Args.PadRight };
        // var ceilMode = pooling.Args.PadMode == 0 ? 1 : 0;
        long countIncludePad = pooling.Args.AvgPoolCountIncludePad ? 1 : 0;

        return (pooling.Args.PoolingType switch
        {
            0 => OrtKI.MaxPool(input, "NOTSET", pooling.Args.CeilMode?1:0, dilation, kernelSize, padData, countIncludePad, stride)[0],
            1 => OrtKI.AveragePool(input, "NOTSET", pooling.Args.CeilMode?1:0, countIncludePad, kernelSize, padData, stride),
            _ => throw
                new NotImplementedException($"Pooling type {pooling.Args.PoolingType} is not supported."),
        }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnPooling target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnPooling.Input);
        return Visit(target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnPooling target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnPooling.Input);
        var kernelSize = new int[] { target.Args.KernelH, target.Args.KernelW };
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType, kernelSize[0] * kernelSize[1]),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnPooling target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnPooling.Input);
        var returnType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(returnType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(inputType) * (MetricUtility.DivFLOPs + MetricUtility.ExpFLOPs + MetricUtility.MulFLOPs + MetricUtility.SubFLOPs + MetricUtility.AddFLOPs + (MetricUtility.CmpFLOPs * 2)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnPooling target) => context.GetArgumentShape(target, NcnnPooling.Input);

    private IRType Visit(NcnnPooling pooling, TensorType input)
    {
        // var input = context.GetOrtArgumentValue(pooling, NcnnPooling.Input);
        var kernelSize = new long[] { pooling.Args.KernelH, pooling.Args.KernelW };
        var stride = new long[] { pooling.Args.StrideH, pooling.Args.StrideW };
        var padData = new long[] { pooling.Args.PadTop, pooling.Args.PadBottom, pooling.Args.PadLeft, pooling.Args.PadRight };

        long countIncludePad = pooling.Args.AvgPoolCountIncludePad ? 1 : 0;
        var newInput = new TensorType(input.DType, input.Shape.InsertAndClone(0, 1));
        var output_ = TypeInference.ReduceWindow2DType(newInput, kernelSize, stride,
            Tensor.From(padData, new[] { 2, 2, }), pooling.Args.CeilMode);
        if (output_ is TensorType t)
        {
            var newShape = t.Shape.ToArray();
            return new TensorType(t.DType, newShape[1..]);
        }

        return output_;
    }
}
