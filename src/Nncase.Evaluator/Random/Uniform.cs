// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Random;
using OrtKISharp;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="Uniform"/>.
/// </summary>
public class UniformEvaluator : IEvaluator<Uniform>, ITypeInferencer<Uniform>, ICostEvaluator<Uniform>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Uniform random)
    {
        var shape = context.GetArgumentValueAsArray<long>(random, Uniform.Shape);
        var high = context.GetArgumentValueAsScalar<float>(random, Uniform.High);
        var low = context.GetArgumentValueAsScalar<float>(random, Uniform.Low);
        var seed = context.GetArgumentValueAsScalar<int>(random, Uniform.Seed);

        // 1 is float, onnx only support float/half/double
        var t = OrtKI.RandomUniform(1, high, low, seed, shape);
        return Value.FromTensor(t.ToTensor().CastTo(random.Type));
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

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Uniform target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }
}
