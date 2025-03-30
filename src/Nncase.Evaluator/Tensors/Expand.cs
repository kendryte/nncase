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
public sealed partial class ExpandEvaluator : IEvaluator<Expand>, ITypeInferencer<Expand>, ICostEvaluator<Expand>, IMetricEvaluator<Expand>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Expand expand)
    {
        var input = context.GetArgumentValueAsTensor(expand, Expand.Input);
        var originType = input.ElementType;
        if (originType.IsFloat() && originType != DataTypes.Float32)
        {
            input = input.CastTo(DataTypes.Float32);
        }

        var inputOrt = input.ToOrtTensor();
        var shape = context.GetInt64OrtTensorArgumentValue(expand, Expand.Shape);
        return OrtKI.Expand(inputOrt, shape).ToValue(originType);
    }

    public Cost Visit(ICostEvaluateContext context, Expand target)
    {
        var input = context.GetArgumentType<IRType>(target, Expand.Input);
        var ret = context.GetReturnType<IRType>();

        return CostUtility.GetBroadcastCost(input, ret);
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
        var shape = context.CheckArgumentTensorTypeOrBroadcast(target, Expand.Shape);
        return input switch
        {
            TensorType t => Visit(context, target, t, shape),
            DistributedType d => Visit(context, target, d, shape),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    private IRType Visit(ITypeInferenceContext context, Expand target, TensorType input, TensorType shape)
    {
        var shapeExpr = context.GetArgument(target, Expand.Shape);
        var newShape = TypeInference.ExpandShape(input.Shape, Shape.FromExpr(shapeExpr));
        return input with { Shape = newShape };
    }

    private IRType Visit(ITypeInferenceContext context, Expand target, DistributedType input, TensorType shape)
    {
        var invalid = new InvalidType(input.ToString());
        var shapeExpr = Shape.FromExpr(context.GetArgument(target, Expand.Shape));
        if (input.TensorType.Shape.IsRanked && shapeExpr.IsRanked)
        {
            var newShape = TypeInference.ExpandShape(input.TensorType.Shape, shapeExpr);
            var ndsbp = new SBP[newShape.Count];
            for (int i = 0; i < ndsbp.Length; i++)
            {
                if (input.AxisPolices[i] is SBPSplit && newShape[i] != input.TensorType.Shape[i])
                {
                    return invalid;
                }

                ndsbp[i] = input.AxisPolices[i];
            }

            return new DistributedType(new TensorType(input.TensorType.DType, newShape), ndsbp, input.Placement);
        }

        return invalid;
    }
}
