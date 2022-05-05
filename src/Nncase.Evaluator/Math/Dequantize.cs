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
        return input.DataType switch
        {
            OrtDataType.Int8 => OrtKI.DequantizeLinear(input,
                dequantParam.Scale, 
                (sbyte) dequantParam.ZeroPoint,
                0).ToValue(),
            OrtDataType.UInt8 => OrtKI.DequantizeLinear(input,
                dequantParam.Scale, 
                (byte) dequantParam.ZeroPoint,
                0).ToValue(),
            _ => throw new NotImplementedException("Deq only impl qint8")
        };
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
