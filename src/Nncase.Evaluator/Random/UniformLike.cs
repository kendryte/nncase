// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Random;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="UniformLike"/>.
/// </summary>
public class UniformLikeEvaluator : IEvaluator<UniformLike>, ITypeInferencer<UniformLike>
{
    /// <inheritdoc/>
    public Const Visit(EvaluatorContext context, UniformLike random)
    {
        throw new NotImplementedException();
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
