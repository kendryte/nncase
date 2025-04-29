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
public class ShapeOfEvaluator : IEvaluator<ShapeOf>, ITypeInferencer<ShapeOf>, ICostEvaluator<ShapeOf>, IMetricEvaluator<ShapeOf>
{
    public IValue Visit(IEvaluateContext context, ShapeOf shape)
    {
        var input = context.GetArgumentValueAsTensor(shape, ShapeOf.Input);
        var shapeArr = input.Shape.Select(x => (long)x.FixedValue).ToArray();
        return Value.FromTensor(Tensor.From<long>(shapeArr));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ShapeOf target)
    {
        var input = context.CheckArgumentType<IRType>(target, ShapeOf.Input);
        return input switch
        {
            TensorType t => Visit(context, target, t),
            DistributedType d => Visit(context, target, d),
            AnyType => AnyType.Default,
            _ => new InvalidType(input.GetType().Name),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, ShapeOf target)
    {
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, ShapeOf target)
    {
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    private IRType Visit(ITypeInferenceContext context, ShapeOf target, TensorType input)
    {
        var inExpr = context.GetArgument(target, ShapeOf.Input);
        if (inExpr is TensorConst || input.Shape.IsRanked)
        {
            return new TensorType(DataTypes.Int64, new RankedShape(input.Shape.Rank));
        }

        return new TensorType(DataTypes.Int64, Shape.Unknown(1));
    }

    private IRType Visit(ITypeInferenceContext context, ShapeOf target, DistributedType input)
    {
        var outType = Visit(context, target, input.TensorType);
        if (outType is not TensorType tensorType)
        {
            return new InvalidType("not support input tensor type infer");
        }

        var ndsbp = Enumerable.Repeat(SBP.B, input.Placement.Rank).ToArray();
        return new DistributedType(tensorType, ndsbp, input.Placement);
    }
}
