// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Dequantize"/>.
/// </summary>
public class DequantizeEvaluator : IEvaluator<Dequantize>, ITypeInferencer<Dequantize>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Dequantize target)
    {
        var input = context.GetOrtArgumentValue(target, Dequantize.Input);
        var scale = context.GetOrtArgumentValue(target, Dequantize.Scale);
        var bias = context.GetOrtArgumentValue(target, Dequantize.ZeroPoint);
        return OrtKI.DequantizeLinear(input, scale, bias, 0).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Dequantize target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Dequantize.Input);
        var zeroPoint = context.CheckArgumentType<TensorType>(target, Dequantize.ZeroPoint);
        var scale = context.CheckArgumentType<TensorType>(target, Dequantize.Scale);
        return Visit(target, input, zeroPoint, scale);
    }

    private IRType Visit(Dequantize target, TensorType input, TensorType zeroPoint, TensorType scale)
    {
        return new TensorType(target.TargetType, input.Shape);
    }
}
