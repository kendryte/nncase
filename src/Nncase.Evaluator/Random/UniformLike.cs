// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Random;
using OrtKISharp;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="UniformLike"/>.
/// </summary>
public class UniformLikeEvaluator : IEvaluator<UniformLike>, ITypeInferencer<UniformLike>, ICostEvaluator<UniformLike>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, UniformLike random)
    {
        var input = context.GetOrtArgumentValue(random, UniformLike.Input);
        var high = context.GetArgumentValueAsScalar<float>(random, UniformLike.High);
        var low = context.GetArgumentValueAsScalar<float>(random, UniformLike.Low);
        var seed = context.GetArgumentValueAsScalar<int>(random, UniformLike.Seed);

        // 1 is float, onnx only support float/half/double
        var t = OrtKI.RandomUniformLike(input, 1, high, low, seed);
        return Value.FromTensor(t.ToTensor().CastTo(random.Type));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, UniformLike target)
    {
        var input = context.CheckArgumentType<TensorType>(target, UniformLike.Input);
        return Visit(target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, UniformLike target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    private IRType Visit(UniformLike target, TensorType input)
    {
        return input with { DType = target.Type };
    }
}
