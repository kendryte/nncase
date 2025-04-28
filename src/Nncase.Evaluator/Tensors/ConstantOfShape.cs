// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="ConstantOfShape"/>.
/// </summary>
public class ConstantOfShapeEvaluator : IEvaluator<ConstantOfShape>, ITypeInferencer<ConstantOfShape>, ICostEvaluator<ConstantOfShape>, IMetricEvaluator<ConstantOfShape>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ConstantOfShape target)
    {
        var shape = context.GetArgumentValueAsArray<long>(target, ConstantOfShape.Shape);
        var value = context.GetArgumentValueAsTensor(target, ConstantOfShape.Value);
        var result = Enumerable.Repeat(value.Cast<float>()[0], shape.Aggregate(1, (i, i1) => i * (int)i1)).ToArray();
        return OrtKI.Cast(Tensor.From<float>(result, shape).ToOrtTensor(), (int)value.ElementType.ToOrtType()).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ConstantOfShape target)
    {
        var value = context.CheckArgumentType<IRType>(target, ConstantOfShape.Value);
        var shape = context.CheckArgumentType<IRType>(target, ConstantOfShape.Shape);

        return (value, shape) switch
        {
            (TensorType v, TensorType s) => Visit(context, target, v, s),
            (DistributedType v, DistributedType s) => Visit(context, target, v, s),
            _ => new InvalidType($"ConstantOfShape with {value} and {shape}"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, ConstantOfShape target)
    {
        _ = context.GetArgumentType<IRType>(target, ConstantOfShape.Shape);
        var ret = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, ConstantOfShape target)
    {
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(ret),
        };
    }

    private IRType Visit(ITypeInferenceContext context, ConstantOfShape target, TensorType value, TensorType shape)
    {
        var shapeExpr = context.GetArgument(target, ConstantOfShape.Shape);
        return new TensorType(value.DType, Shape.FromExpr(shapeExpr));
    }

    private IRType Visit(ITypeInferenceContext context, ConstantOfShape target, DistributedType value, DistributedType shape)
    {
        var shapeExpr = context.GetArgument(target, ConstantOfShape.Shape);
        var outShape = Shape.FromExpr(shapeExpr);
        var ndsbp = Enumerable.Repeat(SBP.B, outShape.Rank).ToArray();
        return new DistributedType(new TensorType(value.TensorType.DType, outShape), ndsbp, shape.Placement);
    }
}
