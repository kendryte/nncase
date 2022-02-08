// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.NN;
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
    public Const Visit(IEvaluateContext context, ReduceWindow2D r)
    {
        var input = context.GetTorchArgumentValue(r, ReduceWindow2D.Input);
        var kernelSize = context.GetArgumentValueAsArray<long>(r, ReduceWindow2D.Filter);
        var stride = context.GetArgumentValueAsArray<long>(r, ReduceWindow2D.Stride);
        var padding = context.GetArgumentValueAsArray<long>(r, ReduceWindow2D.Padding);
        var countIncludePad = context.GetArgumentValueAsScalar<bool>(r, ReduceWindow2D.CountIncludePad);
        var ceilMode = context.GetArgumentValueAsScalar<bool>(r, ReduceWindow2D.CeilMode);
        var afterPad = torchF.pad(input, padding);
        var zeroPadding = new[] { 0L, 0 };
        return (r.ReduceOp switch
        {
            // avg_pool padding can only pad one side
            ReduceOp.Mean => torchF.avg_pool2d(afterPad, kernelSize, stride, zeroPadding, ceilMode, countIncludePad),
            ReduceOp.Max => torchF.max_pool2d(afterPad, kernelSize, stride, zeroPadding, new[] { 1L, 1 }, ceilMode),
            _ => throw new ArgumentOutOfRangeException(nameof(r.ReduceOp)),
        }).ToConst();
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
