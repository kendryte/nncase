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
        var value = context.CheckArgumentType<TensorType>(target, ConstantOfShape.Value);
        var shape = context.GetArgument(target, ConstantOfShape.Shape);
        var type = value.DType;
        return new TensorType(type, Shape.FromExpr(shape));
    }

    public Cost Visit(ICostEvaluateContext context, ConstantOfShape target)
    {
        _ = context.GetArgumentType<TensorType>(target, ConstantOfShape.Shape);
        var ret = context.GetReturnType<TensorType>();
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
}
