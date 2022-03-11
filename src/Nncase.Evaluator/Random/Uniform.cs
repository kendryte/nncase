// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Random;
using OrtKISharp;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="Uniform"/>.
/// </summary>
public class UniformEvaluator : IEvaluator<Uniform>, ITypeInferencer<Uniform>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Uniform random)
    {
        var shape = context.GetArgumentValueAsArray<long>(random, Uniform.Shape);
        var high = context.GetArgumentValueAsScalar<float>(random, Uniform.High);
        var low = context.GetArgumentValueAsScalar<float>(random, Uniform.Low);
        var seed = context.GetArgumentValueAsScalar<int>(random, Uniform.Seed);
        return OrtKI.RandomUniform((int)random.Type.ToOrtType(), high, low, seed, shape).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Uniform target)
    {
        if (context.GetArgument(target, Uniform.Shape) is TensorConst shapeValue)
        {
            return new TensorType(target.Type, new Shape(shapeValue.Value.Cast<int>()));
        }
        else
        {
            return new TensorType(target.Type, Shape.Unranked);
        }
    }
}
