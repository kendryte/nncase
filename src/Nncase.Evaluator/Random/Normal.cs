// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Random;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="Normal"/>.
/// </summary>
public class NormalEvaluator : IEvaluator<Normal>, ITypeInferencer<Normal>
{
    /// <inheritdoc/>
    public Const Visit(EvaluatorContext context, Normal random)
    {
        var shape = context.GetArgumentConst(random, Normal.Shape).ToArray<int>();
        var mean = context.GetArgumentConstScalar<float>(random, Normal.Mean);
        var scale = context.GetArgumentConstScalar<float>(random, Normal.Scale);
        var seed = context.GetArgumentConstScalar<int>(random, Normal.Seed);
        return tf.random.normal(shape, mean, stddev: scale, seed: seed).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Normal target)
    {
        if (context.GetArgument(target, Normal.Shape) is Const shapeValue)
        {
            return new TensorType(target.Type, new Shape(shapeValue.ToArray<int>()));
        }
        else
        {
            return new TensorType(target.Type, Shape.Unranked);
        }
    }
}
