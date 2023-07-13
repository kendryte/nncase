// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;
using Shape = Nncase.IR.Shape;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Expand"/>.
/// </summary>
[TypeInferGenerator]
public sealed partial class ExpandEvaluator : IEvaluator<Expand>, ITypeInferencer<Expand>, ICostEvaluator<Expand>, IShapeEvaluator<Expand>, IMetricEvaluator<Expand>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Expand expand)
    {
        var input = context.GetOrtArgumentValue(expand, Expand.Input);
        var shape = context.GetInt64OrtTensorArgumentValue(expand, Expand.Shape);
        return OrtKI.Expand(input, shape).ToValue();
    }

    public Cost Visit(ICostEvaluateContext context, Expand target)
    {
        var input = context.GetArgumentType<TensorType>(target, Expand.Input);
        var ret = context.GetReturnType<TensorType>();

        return CostUtility.GetBroadcastCost(input, ret);
    }

    public Expr Visit(IShapeEvaluateContext context, Expand target)
    {
        var shape = context.GetArgument(target, Expand.Shape);
        return shape;
    }

    public Metric Visit(IMetricEvaluateContext context, Expand target)
    {
        var input = context.GetArgumentType<TensorType>(target, Expand.Input);
        var ret = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(input) + CostUtility.GetMemoryAccess(ret),
        };
    }

    private IRType Visit(ITypeInferenceContext context, Expand target, TensorType input, TensorType shape)
    {
        var shape_expr = context.GetArgument(target, Expand.Shape);
        if (shape_expr is TensorConst constShape)
        {
            return input with { Shape = new Shape(constShape.Value.Cast<int>()) };
        }
        else
        {
            return input with { Shape = TypeInference.ReshapeTo(shape) };
        }
    }
}
