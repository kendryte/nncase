// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Quantize"/>.
/// </summary>
public class QuantizeEvaluator : IEvaluator<Quantize>, ITypeInferencer<Quantize>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Quantize target)
    {
        throw new NotImplementedException();
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
