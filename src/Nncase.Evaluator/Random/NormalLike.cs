// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Random;
using OrtKISharp;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="NormalLike"/>.
/// </summary>
public class NormalLikeEvaluator : IEvaluator<NormalLike>, ITypeInferencer<NormalLike>, ICostEvaluator<NormalLike>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NormalLike random)
    {
        var input = context.GetOrtArgumentValue(random, NormalLike.Input);
        var mean = context.GetArgumentValueAsScalar<float>(random, NormalLike.Mean);
        var scale = context.GetArgumentValueAsScalar<float>(random, NormalLike.Scale);
        var seed = context.GetArgumentValueAsScalar<float>(random, NormalLike.Seed);

        // 1 is float, onnx only support float/half/double
        var t = OrtKI.RandomNormalLike(input, 1, mean, scale, seed);
        return Value.FromTensor(t.ToTensor().CastTo(random.Type));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NormalLike target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NormalLike.Input);
        return Visit(target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NormalLike target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    private IRType Visit(NormalLike target, TensorType input)
    {
        return input with { DType = target.Type };
    }
}
