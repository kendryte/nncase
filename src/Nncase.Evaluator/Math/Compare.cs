// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using TorchSharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Compare"/>.
/// </summary>
public class CompareEvaluator : IEvaluator<Compare>, ITypeInferencer<Compare>
{
    /// <inheritdoc />
    public Const Visit(EvaluatorContext context, Compare target)
    {
        throw new NotImplementedException();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Compare target)
    {
        var lhs = context.CheckArgumentType<TensorType>(target, Binary.Lhs);
        var rhs = context.CheckArgumentType<TensorType>(target, Binary.Rhs);
        return Visit(lhs, rhs);
    }

    private IRType Visit(TensorType lhs, TensorType rhs)
    {
        return TypeInference.BroadcastType(lhs, rhs);
    }
}
