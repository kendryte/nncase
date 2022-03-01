// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;
using TorchSharp;
using static Tensorflow.Binding;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="ReduceWindow2D"/>.
/// </summary>
public class ReduceWindow2DEvaluator : IEvaluator<ReduceWindow2D>, ITypeInferencer<ReduceWindow2D>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ReduceWindow2D r)
    {
        var input = context.GetOrtArgumentValue(r, ReduceWindow2D.Input);
        var kernelSize = context.GetArgumentValueAsArray<long>(r, ReduceWindow2D.Filter);
        var stride = context.GetArgumentValueAsArray<long>(r, ReduceWindow2D.Stride);
        var dilation = context.GetArgumentValueAsArray<long>(r, ReduceWindow2D.Dilation);
        var pads = context.GetArgumentValueAsArray<long>(r, ReduceWindow2D.Padding);
        var countIncludePad = context.GetArgumentValueAsScalar<long>(r, ReduceWindow2D.CountIncludePad);
        var ceilMode = context.GetArgumentValueAsScalar<long>(r, ReduceWindow2D.CeilMode);
        return (r.ReduceOp switch
        {
            ReduceOp.Mean => OrtKI.AveragePool(input, "NOTSET", ceilMode, countIncludePad, kernelSize, pads, stride),
            ReduceOp.Max => OrtKI.MaxPool(input, "NOTSET", ceilMode, dilation, kernelSize, pads, countIncludePad, stride)[0],
            _ => throw new ArgumentOutOfRangeException(nameof(r.ReduceOp)),
        }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ReduceWindow2D target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ReduceWindow2D.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, ReduceWindow2D target, TensorType input)
    {
        var args = context.GetArguments(target, ReduceWindow2D.Filter, ReduceWindow2D.Stride, ReduceWindow2D.Padding, ReduceWindow2D.CeilMode);
        return TypeInference.ReduceWindow2DType(input, args[0], args[1], args[2], args[3]);
    }
}
