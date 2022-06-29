// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Tile"/>.
/// </summary>
public class TileEvaluator : IEvaluator<Tile>, ITypeInferencer<Tile>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Tile tile)
    {
        var input = context.GetOrtArgumentValue(tile, Tile.Input);
        var repeats = context.GetOrtArgumentValue(tile, Tile.Repeats);
        return OrtKI.Tile(input, repeats).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Tile target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Tile.Input);
        var repeat = context.CheckArgumentType<TensorType>(target, Tile.Repeats);
        return Visit(context, target, input, repeat);
    }

    private IRType Visit(ITypeInferenceContext context, Tile target, TensorType input, TensorType repeat)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }
        if (context.GetArgument(target, Tile.Repeats) is TensorConst repeats && input.Shape.IsFixed)
        {
            var shape = OrtKI.Mul(input.Shape.ToValueArray(), repeats.Value.ToArray<int>());
            return input with { Shape = new Shape(shape.ToArray<int>()) };
        }
        return new TensorType(input.DType,
            new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Rank)));
    }
}
