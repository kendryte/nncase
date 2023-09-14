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
        var axis = gather.Axis;
        var index = context.GetOrtArgumentValue(gather, Gather.Index);
        return OrtKI.Gather(input, index, axis).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Gather target)
    {
        var input = context.CheckArgumentType<IRType>(target, Gather.Input);
        var index = context.CheckArgumentType<IRType>(target, Gather.Index);

        return (input, index) switch
        {
            (TensorType a, TensorType b) => Visit(a, target.Axis, b),
            (DistributedType a, DistributedType b) => Visit(a, target.Axis, b),
            _ => new InvalidType($"{input}, {index}"),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Gather target)
    {
        var inputType = context.GetArgumentType<IRType>(target, Gather.Input);
        var indexType = context.GetArgumentType<IRType>(target, Gather.Index);
        var retType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(indexType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(retType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(retType),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Gather target)
    {
        var inShape = context.GetArgumentShape(target, Gather.Input);
        var axis = ShapeExprUtility.Positive(target.Axis, inShape);
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

    private IRType Visit(TensorType input, int axis, TensorType index)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        axis = axis < 0 ? axis + input.Shape.Rank : axis;

        // input_shape[:axis] + index_shape + input_shape[axis + 1:]
        var inShape = input.Shape.ToArray();
        var newShape = inShape[..axis].Concat(index.Shape).Concat(inShape[(axis + 1)..]).ToArray();
        return new TensorType(input.DType, newShape);
    }

    private IRType Visit(DistributedType input, int axis, DistributedType index)
    {
        var invalid = new InvalidType(input.ToString() + " " + index.ToString());
        if (Visit(input.TensorType, axis, index.TensorType) is not TensorType tensorType)
        {
            return invalid;
        }

        if (input.Placement != index.Placement)
        {
            return invalid;
        }

        var ndsbp = new SBP[input.Placement.Rank];

        for (int i = 0; i < input.Placement.Rank; i++)
        {
            switch (input.NdSbp[i], index.NdSbp[i])
            {
                case (SBPSplit { Axis: int ix }, _) when ix == axis:
                    return new InvalidType($"the input can't split on {axis}");
                case (SBPBroadCast, SBPSplit { Axis: int ix }):
                    ndsbp[i] = SBP.S(ix);
                    break;
                case (SBPSplit { Axis: int ix }, SBPBroadCast):
                    ndsbp[i] = SBP.S(ix - axis + index.TensorType.Shape.Rank - 1);
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(tensorType, ndsbp, input.Placement);
    }
}
