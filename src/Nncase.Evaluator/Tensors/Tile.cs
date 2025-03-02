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
        var repeat = context.CheckArgumentType<TensorType>(target, Tile.Repeats);
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

    private IRType Visit(ITypeInferenceContext context, Tile target, TensorType input, TensorType repeat)
    {
        var inShape = input.Shape;
        var repeats = context.GetArgument(target, Tile.Repeats);
        if (repeats is TensorConst tc)
        {
            var repeatsValue = tc.Value.ToArray<int>();
            var shape = input.Shape.Zip(repeatsValue).Select(p => p.First * p.Second);
            return input with { Shape = new Shape(shape) };
        }
        else
        {
            var shape = input.Shape.Select((p, i) => p * (Dimension)repeats[i]);
            return input with { Shape = new Shape(shape) };
        }
    }
}
