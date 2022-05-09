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
        var quantParam = context.GetArgumentValueAsScalar<QuantParam>(target, Quantize.QuantParam);
        var zeroPoint = Tensor.FromScalar(quantParam.ZeroPoint).CastTo(target.TargetType);
        return OrtKI.QuantizeLinear(input, quantParam.Scale, zeroPoint.ToOrtTensor(), 0).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Quantize target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Quantize.Input);
        var quantParam = context.CheckArgumentType<TensorType>(target, Quantize.QuantParam);
        return Visit(target, input, quantParam);
    }

    private IRType Visit(Quantize target, TensorType input, TensorType quantParam)
    {
        return new TensorType(target.TargetType, input.Shape);
    }
}
