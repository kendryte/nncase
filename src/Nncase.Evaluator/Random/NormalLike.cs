// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Random;
using OrtKISharp;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="NormalLike"/>.
/// </summary>
public class NormalLikeEvaluator : IEvaluator<NormalLike>, ITypeInferencer<NormalLike>
{ 
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NormalLike random)
    {
        var input = context.GetOrtArgumentValue(random, NormalLike.Input);
        var mean = context.GetArgumentValueAsScalar<float>(random, NormalLike.Mean);
        var scale = context.GetArgumentValueAsScalar<float>(random, NormalLike.Scale);
        var seed = context.GetArgumentValueAsScalar<float>(random, NormalLike.Seed);
        return OrtKI.RandomNormalLike(input, (int)random.Type.ToOrtType(), mean, scale, seed).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NormalLike target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NormalLike.Input);
        return Visit(target, input);
    }

    private IRType Visit(NormalLike target, TensorType input)
    {
        return input with { DType = target.Type };
    }
}
