// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="RangeOf"/>.
/// </summary>
public class RangeOfEvaluator : IEvaluator<RangeOf>, ITypeInferencer<RangeOf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, RangeOf target)
    {
        var input = context.GetArgumentValueAsTensor<float>(target, RangeOf.Input);
        var min = float.MaxValue;
        var max = float.MinValue;
        foreach (var f in input)
        {
            if (!(f == float.NaN || float.IsInfinity(f)))
            {
                min = System.Math.Min(min, f);
                max = System.Math.Max(max, f);
            }
        }

        return Value.FromTensor(new[] {min, max});
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, RangeOf target)
    {
        var input = context.CheckArgumentType<TensorType>(target, RangeOf.Input);
        return input with { Shape = new Shape(2) };
    }
}