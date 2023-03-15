// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Random;
using OrtKISharp;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Evaluator for <see cref="Normal"/>.
/// </summary>
public class NormalEvaluator : IEvaluator<Normal>, ITypeInferencer<Normal>, ICostEvaluator<Normal>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Normal random)
    {
        var shape = context.GetArgumentValueAsArray<long>(random, Normal.Shape);
        var mean = context.GetArgumentValueAsScalar<float>(random, Normal.Mean);
        var scale = context.GetArgumentValueAsScalar<float>(random, Normal.Scale);
        var seed = context.GetArgumentValueAsScalar<float>(random, Normal.Seed);

        // 1 is float, onnx only support float/half/double
        if (shape.Length == 0)
        {
            var f = OrtKI.RandomNormal(1, mean, scale, seed, new long[] { 1 }).ToArray<float>()[0];
            return Value.FromTensor(Tensor.FromScalar(f).CastTo(random.Type));
        }

        var t = OrtKI.RandomNormal(1, mean, scale, seed, shape);
        return Value.FromTensor(t.ToTensor().CastTo(random.Type));
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

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Normal target)
    {
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }
}
