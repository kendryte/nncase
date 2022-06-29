// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="ShapeOf"/>.
/// </summary>
public class ShapeOfEvaluator : IEvaluator<ShapeOf>, ITypeInferencer<ShapeOf>, ICostEvaluator<ShapeOf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ShapeOf shape)
    {
        var input = context.GetArgumentValueAsTensor(shape, ShapeOf.Input);
        return Value.FromTensor(Tensor.FromSpan(input.Dimensions));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ShapeOf target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ShapeOf.Input);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost? Visit(ICostEvaluateContext context, ShapeOf target)
    {
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    private IRType Visit(ITypeInferenceContext context, ShapeOf target, TensorType input)
    {
        return new TensorType(DataTypes.Int64, new Shape(input.Shape.Rank));
    }
}
