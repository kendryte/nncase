// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Quantize"/>.
/// </summary>
public class QuantizeEvaluator : IEvaluator<Quantize>, ITypeInferencer<Quantize>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Quantize target)
    {
        var input = context.GetOrtArgumentValue(target, Quantize.Input);
        var scale = context.GetOrtArgumentValue(target, Quantize.Scale);
        var bias = context.GetOrtArgumentValue(target, Quantize.ZeroPoint);
        return OrtKI.QuantizeLinear(input, scale, bias, 0).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Quantize target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Dequantize.Input);
        var zeroPoint = context.CheckArgumentType<TensorType>(target, Dequantize.ZeroPoint);
        var scale = context.CheckArgumentType<TensorType>(target, Dequantize.Scale);
        return Visit(target, input, zeroPoint, scale);
    }

    private IRType Visit(Quantize target, TensorType input, TensorType zeroPoint, TensorType scale)
    {
        return new TensorType(target.TargetType, input.Shape);
    }
}
