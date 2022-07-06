// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="ConstantOfShape"/>.
/// </summary>
public class ConstantOfShapeEvaluator : IEvaluator<ConstantOfShape>, ITypeInferencer<ConstantOfShape>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ConstantOfShape target)
    {
        var shape = context.GetArgumentValueAsArray<int>(target, ConstantOfShape.Shape);
        var value = context.GetArgumentValueAsTensor(target, ConstantOfShape.Value);
        var result = Enumerable.Repeat(value.ToScalar<float>(), shape.Aggregate(1, (i, i1) => i * i1)).ToArray();
        return OrtKI.Cast(Tensor.FromSpan<float>(result, shape).ToOrtTensor(), (int) value.ElementType.ToOrtType()).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ConstantOfShape target)
    {
        var shape = context.GetArgument(target, ConstantOfShape.Shape);
        var value = context.CheckArgumentType<TensorType>(target, ConstantOfShape.Value);
        var type = value.DType;
        if (shape is TensorConst shapeValue)
        {
            return new TensorType(type, shapeValue.Value.ToArray<int>());
        }
        else
        {
            return new TensorType(type, Shape.Unranked);
        }
    }
}
