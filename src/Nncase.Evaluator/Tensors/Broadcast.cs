// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Broadcast"/>.
/// </summary>
[TypeInferGenerator]
public sealed partial class BroadcastEvaluator : IEvaluator<Broadcast>, ITypeInferencer<Broadcast>, ICostEvaluator<Broadcast>, IMetricEvaluator<Broadcast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Broadcast b)
    {
        var input = context.GetOrtArgumentValue(b, Broadcast.Input);
        var shape = context.GetArgumentValueAsArray<long>(b, Broadcast.Shape);
        if (input.DataType == OrtDataType.Bool)
        {
            return input.Cast(OrtDataType.Float).BroadcastTo(shape).Cast(OrtDataType.Bool).ToValue();
        }

        return OrtKIExtensions.BroadcastTo(input, shape, input.DataType).ToValue();
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Broadcast target)
    {
        var input = context.GetArgumentType<TensorType>(target, Broadcast.Input);
        var ret = context.GetReturnType<TensorType>();
        return CostUtility.GetBroadcastCost(input, ret);
    }

    public Metric Visit(IMetricEvaluateContext context, Broadcast target)
    {
        var input = context.GetArgumentType<TensorType>(target, Broadcast.Input);
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(input) + CostUtility.GetMemoryAccess(ret),
        };
    }

    private IRType Visit(TensorType input, TensorType shape, ITypeInferenceContext context, Broadcast op)
    {
        var shapeValue = context.GetArgument(op, Broadcast.Shape);
        if (shapeValue is TensorConst constShapeValue && input.Shape.IsFixed)
        {
            return TypeInference.BroadcastType(input, new TensorType(input.DType, constShapeValue.Value.ToArray<int>()));
        }

        if (shape.Shape[0].IsFixed)
        {
            return input with { Shape = Enumerable.Repeat(Dimension.Unknown, shape.Shape[0].FixedValue).ToArray() };
        }

        return input with { Shape = IR.Shape.Unranked };
    }
}
