// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Random;
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
        throw new NotImplementedException();
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
