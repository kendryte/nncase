// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="CumSum"/>.
/// </summary>
public class CumSumEvaluator : IEvaluator<CumSum>, ITypeInferencer<CumSum>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, CumSum cumSum)
    {
        var input = context.GetOrtArgumentValue(cumSum, CumSum.Input);
        // in onnx, CumSum.Axis is a input tensor with one value
        var axis = context.GetOrtArgumentValue(cumSum, CumSum.Axis);
        var exclusive = context.GetArgumentValueAsScalar<long>(cumSum, CumSum.Exclusive);
        var reverse = context.GetArgumentValueAsScalar<long>(cumSum, CumSum.Reverse);
        return OrtKI.CumSum(input, axis, exclusive, reverse).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, CumSum target)
    {
        var input = context.CheckArgumentType<TensorType>(target, CumSum.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
