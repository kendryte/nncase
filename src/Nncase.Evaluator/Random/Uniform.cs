// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Random;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="Uniform"/>.
/// </summary>
public class UniformEvaluator : IEvaluator<Uniform>, ITypeInferencer<Uniform>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Uniform random)
    {
        var shape = context.GetArgumentValue(random, Normal.Shape).ToArray<int>();
        var mean = context.GetArgumentValueAsScalar<float>(random, Normal.Mean);
        var scale = context.GetArgumentValueAsScalar<float>(random, Normal.Scale);
        var seed = context.GetArgumentValueAsScalar<int>(random, Normal.Seed);
        return tf.random.normal(shape, mean, stddev: scale, seed: seed).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Uniform target)
    {
        if (context.GetArgument(target, Uniform.Shape) is Const shapeValue)
        {
            return new TensorType(target.Type, new Shape(shapeValue.ToArray<int>()));
        }
        else
        {
            return new TensorType(target.Type, Shape.Unranked);
        }
    }
}
