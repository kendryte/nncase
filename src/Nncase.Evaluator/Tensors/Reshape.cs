// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using Range = Nncase.IR.Tensors.Range;
using Reshape = Nncase.IR.Tensors.Reshape;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Range"/>.
/// </summary>
public class ReshapeEvaluator : IEvaluator<Reshape>, ITypeInferencer<Reshape>, ICostEvaluator<Reshape>, IMetricEvaluator<Reshape>
{
    public static IRType VisitDistributedType(DistributedType inType, RankedShape newShape)
    {
        var invalidType = new InvalidType($"not supported reshape {inType} to {newShape}");
        var inShape = (RankedShape)inType.TensorType.Shape;
        var maxInShape = CompilerServices.GetMaxShape(inShape);
        var maxNewShape = CompilerServices.GetMaxShape(newShape);
        if (!IRUtility.TryGetShapeMapMatrix(maxInShape, maxNewShape, out var mat))
        {
            return invalidType;
        }

        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsCompleteDict(mat);
        var newAxisPolicies = new SBP[newShape.Rank];

        // 1. [1024@t] -> [8@t, 128]
        foreach ((var inAxis, var newAxes) in forwardDict)
        {
            var inPolicy = inType.AxisPolicies[inAxis];
            if (inPolicy is SBPSplit split)
            {
                var newAxesOffset = newAxes[0];
                var newDims = newAxes.Select(newAxis => newShape[newAxis]).ToArray().AsReadOnlySpan();
                var newSplitAxis = newDims.FirstIndexOfNotEqual(1);
                newSplitAxis = newSplitAxis < 0 ? 0 : newSplitAxis;
                if (newDims[newSplitAxis] != inShape[inAxis]
                    && !Dimension.TryDivExactly(newDims[newSplitAxis], DistributedUtility.GetDivisor(split, inType.Placement), out _))
                {
                    return invalidType;
                }

                foreach (var newAxis in newAxes)
                {
                    newAxisPolicies[newAxis] = newAxis == (newAxesOffset + newSplitAxis) ? split : SBP.B;
                }
            }
            else
            {
                foreach (var newAxis in newAxes)
                {
                    newAxisPolicies[newAxis] = inPolicy;
                }
            }
        }

        // 2. [8@t, 128] -> [1024@t]
        foreach ((var newAxis, var inAxes) in backwardDict)
        {
            if (newAxisPolicies[newAxis] is not null)
            {
                continue; // already set
            }

            var splitAxes = from inAxis in inAxes
                            let inPolicy = inType.AxisPolicies[inAxis]
                            where inPolicy is SBPSplit
                            select (int?)inAxis;
            if (splitAxes.Count() > 1)
            {
                return invalidType; // more than one split axis, cannot reshape
            }

            var firstSplitAxis = splitAxes.FirstOrDefault();
            if (firstSplitAxis is not null)
            {
                // Either the axis is the first axis or all of the dimensions before it are 1.
                if (firstSplitAxis != inAxes[0]
                    && inAxes.TakeWhile(a => a < firstSplitAxis).Any(a => inShape[a] != 1))
                {
                    return invalidType;
                }

                newAxisPolicies[newAxis] = inType.AxisPolicies[firstSplitAxis.Value];
            }
            else
            {
                newAxisPolicies[newAxis] = SBP.B; // no split axis, use B
            }
        }

        if (newAxisPolicies.Any(a => a is null))
        {
            if (inType.AxisPolicies.Select((x, i) => (x, i)).Where(x => !forwardDict.ContainsKey(x.i)).All(x => x.x is SBPBroadCast))
            {
                // If all axes that are not in the forward mapping are broadcast, we can still reshape.
                for (int i = 0; i < newAxisPolicies.Length; i++)
                {
                    if (newAxisPolicies[i] is null)
                    {
                        newAxisPolicies[i] = SBP.B;
                    }
                }
            }
            else
            {
                // If there are axes that are not in the forward mapping and they are not broadcast, we cannot reshape.
                return invalidType;
            }
        }

        return new DistributedType(inType.TensorType with { Shape = newShape }, newAxisPolicies, inType.Placement);
    }

    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Reshape reshape)
    {
        OrtKISharp.Tensor input;

        var inputOrg = context.GetArgumentValue(reshape, Reshape.Input).AsTensor();
        var dataType = inputOrg.ElementType;
        if (dataType is VectorType { ElemType: DataType dataTypes } vType && dataTypes != DataTypes.Float32)
        {
            var interType = new VectorType(DataTypes.Float32, vType.Lanes);
            input = Nncase.IR.F.Tensors.Cast(inputOrg, interType).Evaluate().AsTensor().ToOrtTensor();
        }
        else if (dataType is not VectorType && dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            input = Nncase.IR.F.Tensors.Cast(inputOrg, DataTypes.Float32).Evaluate().AsTensor().ToOrtTensor();
        }
        else
        {
            input = context.GetOrtArgumentValue(reshape, Reshape.Input);
        }

        var shape = context.GetArgumentValueAsArray<long>(reshape, Reshape.Shape);
        if (context.CurrentCall.CheckedType is AnyType)
        {
            return Value.FromTensor(OrtKI.Reshape(input, shape, 0).ToTensor());
        }

        var tensorType = context.CurrentCall.CheckedTensorType;
        var allowzero = tensorType.Shape is RankedShape rankedShape && rankedShape.Contains(0) ? 1L : 0L;
        if (tensorType.DType is VectorType vtype)
        {
            shape = shape.Concat(vtype.Lanes.Select(i => (long)i)).ToArray();
        }

        var reshaped = OrtKI.Reshape(input, shape, allowzero);
        if (dataType is not VectorType && dataType != DataTypes.Float32)
        {
            reshaped = OrtKI.Cast(OrtKI.Reshape(input, shape, allowzero), (int)dataType.ToOrtType());
        }

        return reshaped.ToValue(dataType);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Reshape target)
    {
        var input = context.CheckArgumentType<IRType>(target, Reshape.Input);
        return input switch
        {
            TensorType tensorType => Visit(context, target, tensorType),
            DistributedType distributedType => Visit(context, target, distributedType),
            AnyType => AnyType.Default,
            InvalidType => input,
            _ => new InvalidType($"Not Support Input Type {input.GetType().Name}"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, Reshape target)
    {
        return CostUtility.GetReshapeCost();
    }

    Cost ICostEvaluator<Reshape>.Visit(ICostEvaluateContext context, Reshape target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Reshape target)
    {
        return Metric.Zero;
    }

    private IRType Visit(ITypeInferenceContext context, Reshape target, TensorType input)
    {
        var shape = (Shape)context.GetArgument(target, Reshape.Shape);
        var outShape = TypeInference.ReshapeShape(input.Shape, shape);
        return input with { Shape = outShape };
    }

    private IRType Visit(ITypeInferenceContext context, Reshape target, DistributedType inputType)
    {
        var outType = Visit(context, target, inputType.TensorType);
        if (outType is not TensorType outTensorType)
        {
            return outType;
        }

        var invalid = new InvalidType(inputType.ToString());
        if (outTensorType.Shape.IsUnranked)
        {
            return invalid;
        }

        return VisitDistributedType(inputType, (RankedShape)outTensorType.Shape);
    }
}
