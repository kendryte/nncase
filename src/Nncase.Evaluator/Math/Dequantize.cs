// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;
using Tensorflow;

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
        var dequantParam = context.GetArgumentValueAsScalar<QuantParam>(target, Dequantize.DequantParam);
        var zeroPoint = Tensor.FromScalar(-dequantParam.ZeroPoint / dequantParam.Scale).CastTo(input.DataType.ToDataType());
        return OrtKI.DequantizeLinear(input, dequantParam.Scale, zeroPoint.ToOrtTensor(), 0).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Dequantize target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Dequantize.Input);
        var deqParam = context.CheckArgumentType<TensorType>(target, Dequantize.DequantParam);
        return Visit(target, input, deqParam);
    }

    private IRType Visit(Dequantize target, TensorType input, TensorType deqParam)
    {
        return new TensorType(target.TargetType, input.Shape);
    }
}
