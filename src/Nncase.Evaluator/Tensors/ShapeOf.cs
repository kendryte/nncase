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
public class ShapeOfEvaluator : IEvaluator<ShapeOf>, ITypeInferencer<ShapeOf>, ICostEvaluator<ShapeOf>, IShapeEvaluator<ShapeOf>, IMetricEvaluator<ShapeOf>
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
        var input = context.CheckArgumentType<TensorType>(target, ShapeOf.Input);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, ShapeOf target)
    {
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, ShapeOf target)
    {
        return context.GetArgumentShape(target, ShapeOf.Input);
    }

    public Metric Visit(IMetricEvaluateContext context, ShapeOf target)
    {
        var outputType = context.GetReturnType<TensorType>();

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
            return new TensorType(DataTypes.Int64, new Shape(input.Shape.Rank));
        }

        return new TensorType(DataTypes.Int64, new Shape(Dimension.Unknown));
    }
}
