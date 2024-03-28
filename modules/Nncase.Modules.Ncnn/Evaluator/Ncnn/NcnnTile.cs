// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Ncnn;
using OrtKISharp;

namespace Nncase.Evaluator.Ncnn;

/// <summary>
/// Evaluator for <see cref="NcnnTile"/>.
/// </summary>
public class NcnnTileEvaluator : IEvaluator<NcnnTile>, ITypeInferencer<NcnnTile>, ICostEvaluator<NcnnTile>, IShapeEvaluator<NcnnTile>, IMetricEvaluator<NcnnTile>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, NcnnTile tile)
    {
        var input = context.GetOrtArgumentValue(tile, NcnnTile.Input);
        return OrtKI.Tile(input, tile.Repeats).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, NcnnTile target)
    {
        var input = context.CheckArgumentType<TensorType>(target, NcnnTile.Input);
        return Visit(input, target.Repeats);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, NcnnTile target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnTile.Input);
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, NcnnTile target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, NcnnTile.Input);
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(ret),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, NcnnTile target) => context.GetArgumentShape(target, NcnnTile.Input);

    private IRType Visit(TensorType input, int[] repeats)
    {
        var outputShape = input.Shape.ToValueArray();
        for (int i = 0; i < repeats.Length; i++)
        {
            outputShape[i] *= repeats[i];
        }

        return new TensorType(input.DType, outputShape);
    }
}
