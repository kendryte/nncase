// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using Gather = Nncase.IR.Tensors.Gather;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Gather"/>.
/// </summary>
public class GatherEvaluator : IEvaluator<Gather>, ITypeInferencer<Gather>, ICostEvaluator<Gather>, IShapeEvaluator<Gather>, IMetricEvaluator<Gather>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Gather gather)
    {
        var input = context.GetOrtArgumentValue(gather, Gather.Input);
        var axis = context.GetArgumentValueAsScalar<int>(gather, Gather.Axis);
        var index = context.GetOrtArgumentValue(gather, Gather.Index);
        return OrtKI.Gather(input, index, axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Gather target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Gather.Input);
        var axis = context.CheckArgumentType<TensorType>(target, Gather.Axis);
        var index = context.CheckArgumentType<TensorType>(target, Gather.Index);
        return Visit(context, target, input, axis, index);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Gather target)
    {
        var ret_type = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret_type.DType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret_type.DType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret_type.DType, 1),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Gather target)
    {
        var axis = context.GetArgument(target, Gather.Axis);
        var inShape = context.GetArgumentShape(target, Gather.Input);
        axis = ShapeExprUtility.Positive(Cast(axis, DataTypes.Int32), inShape);
        var indexShape = context.GetArgumentShape(target, Gather.Index);
        var outShape = ShapeExprUtility.ReplaceList(inShape, axis, indexShape);
        return outShape;
    }

    public Metric Visit(IMetricEvaluateContext context, Gather target)
    {
        var ret_type = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(ret_type) * 2,
        };
    }

    private IRType Visit(ITypeInferenceContext context, Gather target, TensorType input, TensorType axis, TensorType index)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (context.GetArgument(target, Gather.Axis) is TensorConst axisValue)
        {
            var axisV = axisValue.Value.ToScalar<int>();
            axisV = axisV < 0 ? axisV + input.Shape.Rank : axisV;

            // input_shape[:axis] + index_shape + input_shape[axis + 1:]
            var inShape = input.Shape.ToArray();
            var newShape = inShape[..axisV].Concat(index.Shape).Concat(inShape[(axisV + 1)..]).ToArray();
            return new TensorType(input.DType, newShape);
        }
        else
        {
            return new InvalidType("Gather axis must be constant");
        }
    }
}
