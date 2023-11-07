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
        var input = context.GetArgumentType<IRType>(target, Expand.Input);
        var ret = context.GetReturnType<IRType>();

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

    public IRType Visit(ITypeInferenceContext context, Expand target)
    {
        var input = context.CheckArgumentType<IRType>(target, Expand.Input);
        var shape = context.CheckArgumentType<TensorType>(target, Expand.Shape);
        return input switch
        {
            TensorType t => Visit(context, target, t, shape),
            DistributedType d => Visit(context, target, d, shape),
            _ => new InvalidType(input.GetType().ToString()),
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

    private IRType Visit(ITypeInferenceContext context, Expand target, DistributedType input, TensorType shape)
    {
        var invalid = new InvalidType(input.ToString());
        var shape_expr = context.GetArgument(target, Expand.Shape);
        if (shape_expr is TensorConst constShape)
        {
            var newShape = constShape.Value.ToArray<int>();
            var ndsbp = new SBP[input.Placement.Rank];
            for (int i = 0; i < input.Placement.Rank; i++)
            {
                if (input.NdSBP[i] is SBPSplit sbp && newShape[sbp.Axis] != input.TensorType.Shape[sbp.Axis])
                {
                    return invalid;
                }

                ndsbp[i] = input.NdSBP[i];
            }

            return new DistributedType(new TensorType(input.TensorType.DType, new Shape(newShape)), ndsbp, input.Placement);
        }

        return invalid;
    }
}
