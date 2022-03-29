// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="ShapeOf"/>.
/// </summary>
public class ShapeOfEvaluator : IEvaluator<ShapeOf>, ITypeInferencer<ShapeOf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ShapeOf shape)
    {
        var input = context.GetOrtArgumentValue(shape, ShapeOf.Input);
        return OrtKI.Shape(input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ShapeOf target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ShapeOf.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, ShapeOf target, TensorType input)
    {
        return new TensorType(DataTypes.Int64, new Shape(input.Shape.Rank));
    }
}
