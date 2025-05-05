// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Squeeze"/>.
/// </summary>
public class SqueezeEvaluator : IEvaluator<Squeeze>, ITypeInferencer<Squeeze>, ICostEvaluator<Squeeze>, IMetricEvaluator<Squeeze>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Squeeze squeeze)
    {
        var input = context.GetOrtArgumentValue(squeeze, Squeeze.Input);
        var dims = context.GetInt64OrtTensorArgumentValue(squeeze, Squeeze.Dim);
        return OrtKI.Squeeze(input, dims).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Squeeze target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Squeeze.Input);
        var dims = context.CheckArgumentType<ShapeType>(target, Squeeze.Dim);
        return Visit(context, target, input, dims);
    }

    public Cost Visit(ICostEvaluateContext context, Squeeze target)
    {
        return CostUtility.GetReshapeCost();
    }

    public Metric Visit(IMetricEvaluateContext context, Squeeze target)
    {
        return Metric.Zero;
    }

    private IRType Visit(ITypeInferenceContext context, Squeeze target, TensorType input, ShapeType dims)
    {
        if (input.Shape is not RankedShape inShape)
        {
            return input;
        }

        if (context.GetArgument(target, Squeeze.Dim) is RankedShape { IsFixed: true } dimsValue)
        {
            var outshape = inShape.ToList();
            if (dimsValue.Rank == 0)
            {
                return input with { Shape = new RankedShape(outshape.Where(x => x != 1).ToArray()) };
            }

            foreach (var dimV in dimsValue)
            {
                var dimValue = Dimension.Positive(dimV, inShape.Rank);
                outshape[(int)dimValue.FixedValue] = int.MaxValue;
            }

            return input with { Shape = new RankedShape(outshape.Where(x => x != int.MaxValue)) };
        }
        else if (dims.Rank is int dimsRank)
        {
            return input with { Shape = Shape.Unknown(input.Shape.Rank - dimsRank) };
        }

        return input with { Shape = Shape.Unranked };
    }
}
