// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Hardmax"/>.
/// </summary>
public class HardmaxEvaluator : IEvaluator<Hardmax>, ITypeInferencer<Hardmax>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Hardmax target)
    {
        var input = context.GetOrtArgumentValue(target, Hardmax.Input);
        var axis = context.GetArgumentValueAsScalar<long>(target, Hardmax.Axis);
        return OrtKI.Hardmax(input, axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Hardmax target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Hardmax.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
