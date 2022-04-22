// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;
using SMath = System.Math;
namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="QuantParamOf"/>.
/// </summary>
public class QuantParamOfEvaluator : IEvaluator<QuantParamOf>, ITypeInferencer<QuantParamOf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, QuantParamOf target)
    {
        var rawRange = context.GetArgumentValueAsArray<float>(target, QuantParamOf.Range);
        var bits = context.GetArgumentValueAsScalar<int>(target, QuantParamOf.Bits);
        var min = rawRange[0];
        var max = rawRange[1];
        var range = fixupRange((min, max), target.QuantMode == QuantMode.SignedSymmetricMode);
        double QMax = 255;
        double QMin = 0;
        switch (target.QuantMode)
        {
            case QuantMode.UnsignedMode:
                QMin = 0;
                QMax = (1 << bits) - 1;
                break;
            case QuantMode.SignedSymmetricMode:
                QMin = -(1 << (bits - 1)) + 1;
                QMax = (1 << (bits - 1)) - 1;
                break;
            case QuantMode.SignedAsymmetricMode:
                QMin = -(1 << (bits - 1));
                QMax = (1 << (bits - 1)) - 1;
                break;
            default:
                throw new ArgumentOutOfRangeException("Invalid QuantMode");
        }

        var scale = (range.Max - range.Min) / (QMax - QMin);
        var bias = SMath.Round((range.Min * (QMin - QMax)) / (range.Max - range.Min)) + QMin;
        var r = new QuantParam((float)scale, (int)bias);
        return Value.FromTensor(Tensor.FromScalar(r));
    }
    
    private ValueRange<float> fixupRange(ValueRange<float> range, bool symmetric = false)
    {
        if (symmetric)
        {
            var r = SMath.Max(SMath.Max(SMath.Abs(range.Min), SMath.Abs(range.Max)), 0.01f);
            return (-r, r);
        }
        else
        {
            range.Max = SMath.Max(0, range.Max);
            range.Min = SMath.Min(0, range.Min);
            var r = range.Max - range.Min;
            if (r == 0)
            {
                r = 0.1f;
            }
            else if (r < 0.01f)
            {
                r = 0.01f;
            }

            range.Max = range.Min + r;
        }

        return range;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, QuantParamOf target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Dequantize.Input);
        return new TensorType(new QuantParamType(), Shape.Scalar);
    }
}