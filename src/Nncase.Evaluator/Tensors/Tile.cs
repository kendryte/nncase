// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.TIR;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Tile"/>.
/// </summary>
public class TileEvaluator : IEvaluator<Tile>, ITypeInferencer<Tile>, ICostEvaluator<Tile>, IMetricEvaluator<Tile>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Tile tile)
    {
        var input = context.GetOrtArgumentValue(tile, Tile.Input);
        var repeats = context.GetInt64OrtTensorArgumentValue(tile, Tile.Repeats);
        return OrtKI.Tile(input, repeats).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Tile target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Tile.Input);
        var repeat = context.CheckArgumentType<ShapeType>(target, Tile.Repeats);
        return Visit(context, target, input, repeat);
    }

    public Cost Visit(ICostEvaluateContext context, Tile target)
    {
        var input = context.GetArgumentType<TensorType>(target, Tile.Input);
        var ret = context.GetReturnType<TensorType>();
        return CostUtility.GetBroadcastCost(input, ret);
    }

    public Metric Visit(IMetricEvaluateContext context, Tile target)
    {
        var input = context.GetArgumentType<TensorType>(target, Tile.Input);
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(input) + CostUtility.GetMemoryAccess(ret),
        };
    }

    private IRType Visit(ITypeInferenceContext context, Tile target, TensorType input, ShapeType repeat)
    {
        if (input.Shape is not RankedShape inputShape)
        {
            return input;
        }

        var repeats = (Shape)context.GetArgument(target, Tile.Repeats);
        var shape = inputShape.Select((p, i) => p * repeats[i]);
        return input with { Shape = new RankedShape(shape) };
    }
}
