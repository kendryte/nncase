// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Random;
using OrtKISharp;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="UniformLike"/>.
/// </summary>
public class UniformLikeEvaluator : IEvaluator<UniformLike>, ITypeInferencer<UniformLike>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, UniformLike random)
    {
        var input = context.GetOrtArgumentValue(random, UniformLike.Input);
        var high = context.GetArgumentValueAsScalar<float>(random, UniformLike.High);
        var low = context.GetArgumentValueAsScalar<float>(random, UniformLike.Low);
        var seed = context.GetArgumentValueAsScalar<int>(random, UniformLike.Seed);
        return OrtKI.RandomUniformLike(input, (long)random.Type.ToOrtType(), high, low, seed).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, UniformLike target)
    {
        var input = context.CheckArgumentType<TensorType>(target, UniformLike.Input);
        return Visit(target, input);
    }

    private IRType Visit(UniformLike target, TensorType input)
    {
        return input with { DType = target.Type };
    }
}
