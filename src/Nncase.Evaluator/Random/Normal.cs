// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Random;
using OrtKISharp;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="Normal"/>.
/// </summary>
public class NormalEvaluator : IEvaluator<Normal>, ITypeInferencer<Normal>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Normal random)
    {
        var shape = context.GetArgumentValueAsArray<long>(random, Normal.Shape);
        var mean = context.GetArgumentValueAsScalar<float>(random, Normal.Mean);
        var scale = context.GetArgumentValueAsScalar<float>(random, Normal.Scale);
        var seed = context.GetArgumentValueAsScalar<float>(random, Normal.Seed);
        return OrtKI.RandomNormal((int)random.Type.ToOrtType(), mean, scale, seed, shape).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Normal target)
    {
        if (context.GetArgument(target, Normal.Shape) is TensorConst shapeValue)
        {
            return new TensorType(target.Type, new Shape(shapeValue.Value.Cast<int>()));
        }
        else
        {
            return new TensorType(target.Type, Shape.Unranked);
        }
    }
}
