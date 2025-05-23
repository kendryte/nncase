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
    public static IRType VisitDistributedType(DistributedType inType, Shape newSymbolShape)
    {
        IRType invalidType = new InvalidType($"not supported reshape {inType} to {newSymbolShape}");
        if (inType.TensorType.Shape.IsFixed && newSymbolShape.IsFixed)
        {
            var newShape = newSymbolShape.ToValueArray();
            var inShape = inType.TensorType.Shape.ToValueArray();
            if (!IRUtility.TryGetShapeMapMatrix(inShape, newShape, out var mat))
            {
                return invalidType;
            }

            var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsDict(mat);
            var ndsbpIn = DistributedUtility.AxisPolicesToNDSBP(inType.AxisPolices, inType.Placement.Rank);
            var ndsbp = new SBP[ndsbpIn.Count];
            for (int meshAxis = 0; meshAxis < ndsbp.Length; meshAxis++)
            {
                var inSBP = ndsbpIn[meshAxis];
                switch (inSBP)
                {
                    case SBPSplit si:
                        {
                            var mapedOutAxes = forwardDict[si.Axes[0]];

                            // when input axis is splited-by-reshape, we can direct obtain the tensor which splited-by-sbp on the first maped axis.
                            if (mapedOutAxes.Count > 1)
                            {
                                var firstValidAxis = mapedOutAxes.Where(axis => newShape[axis] > 1).First();
                                var restAxes = mapedOutAxes.Skip(mapedOutAxes.IndexOf(firstValidAxis) + 1).ToArray();
                                var restSize = restAxes.Aggregate(1L, (x, i) => x * newShape[i]);
                                if (restSize < (inShape[si.Axes[0]] / inType.Placement.Hierarchy[meshAxis]))
                                {
                                    ndsbp[meshAxis] = SBP.S(new[] { firstValidAxis });
                                }
                                else
                                {
                                    return invalidType;
                                }
                            }
                            else
                            {
                                // when input axis is merged-by-reshape, we can direct obtain the tensor which splited-by-sbp on the first maped axis.
                                var outAxis = mapedOutAxes.First();

                                // when the outAxis is merged dim, only support no transpose order and no pad.
                                var inAxes = backwardDict[outAxis];
                                if (inAxes.Where(inAxis => inShape[inAxis] != 1).First() == si.Axes[0])
                                {
                                    ndsbp[meshAxis] = SBP.S(new[] { outAxis });
                                }
                                else
                                {
                                    return invalidType;
                                }
                            }
                        }

                        break;
                    default:
                        ndsbp[meshAxis] = inSBP;
                        break;
                }
            }

            var policies = DistributedUtility.NDSBPToAxisPolices(ndsbp, newShape.Length);
            var newTensorType = new TensorType(inType.TensorType.DType, newSymbolShape);
            if (!DistributedUtility.IsDistributable(newTensorType, policies.ToArray(), inType.Placement))
            {
                return invalidType;
            }

            if (policies.ToArray().Count(p => p is SBPSplit) > 1)
            {
                return invalidType;
            }

            return new DistributedType(newTensorType, policies, inType.Placement);
        }
        else
        {
            var inShape = (RankedShape)inType.TensorType.Shape;
            var rankedNewSymbolShape = (RankedShape)newSymbolShape;

            // check is unsequeeze/sequeeze
            if (Enumerable.SequenceEqual(inShape.Where(i => i != 1).ToArray(), rankedNewSymbolShape.Where(i => i != 1).ToArray()))
            {
                if (inShape.Count < rankedNewSymbolShape.Count)
                {
                    var axis = 0;
                    var axisMap = new Dictionary<int, int>();
                    if (!inShape.IsScalar)
                    {
                        for (var n = 0; n < rankedNewSymbolShape.Count; n++)
                        {
                            if (newSymbolShape[n] == inShape[axis])
                            {
                                axisMap.Add(axis++, n);
                                if (axis >= inShape.Count)
                                {
                                    break;
                                }
                            }
                        }
                    }

                    var ndsbpIn = DistributedUtility.AxisPolicesToNDSBP(inType.AxisPolices, inType.Placement.Rank);
                    var ndsbp = new SBP[ndsbpIn.Count];
                    for (int i = 0; i < inType.Placement.Rank; i++)
                    {
                        ndsbp[i] = ndsbpIn[i] switch
                        {
                            SBPSplit { Axes: var sx } => SBPSplit.S(new[] { axisMap[sx[0]] }),
                            SBP sbp => sbp,
                        };
                    }

                    return inType with { TensorType = new TensorType(inType.TensorType.DType, newSymbolShape), AxisPolices = DistributedUtility.NDSBPToAxisPolices(ndsbp, newSymbolShape.Rank) };
                }
                else if (inShape.Count > rankedNewSymbolShape.Count)
                {
                    var axis = 0;
                    var axisMap = new Dictionary<int, int>();
                    for (var o = 0; o < inShape.Count; o++)
                    {
                        if (axis < rankedNewSymbolShape.Count && inShape[o] == newSymbolShape[axis])
                        {
                            axisMap.Add(o, axis++);
                            if (axis >= rankedNewSymbolShape.Count)
                            {
                                break;
                            }
                        }
                    }

                    var ndsbpIn = DistributedUtility.AxisPolicesToNDSBP(inType.AxisPolices, inType.Placement.Rank);
                    var ndsbp = new SBP[ndsbpIn.Count];
                    for (int i = 0; i < inType.Placement.Rank; i++)
                    {
                        ndsbp[i] = ndsbpIn[i] switch
                        {
                            SBPSplit { Axes: var sx } => SBPSplit.S(new[] { axisMap[sx[0]] }),
                            SBP sbp => sbp,
                        };
                    }

                    return inType with { TensorType = new TensorType(inType.TensorType.DType, newSymbolShape), AxisPolices = DistributedUtility.NDSBPToAxisPolices(ndsbp, newSymbolShape.Rank) };
                }
            }

            // not the squeeze or unsqueeze
            if (!inType.AxisPolices.Any(sbp => sbp is SBPSplit))
            {
                return inType with { TensorType = new TensorType(inType.TensorType.DType, newSymbolShape), AxisPolices = Enumerable.Repeat(SBP.B, newSymbolShape.Rank).ToArray() };
            }
        }

        return invalidType;

        // TODO: type infer using nttd.
#if false
        var ndsbp = new SBP[newShape.Length];

        for (int dimAxis = 0; dimAxis < inType.AxisPolices.Count; dimAxis++)
        {
            var inSBP = inType.AxisPolices[dimAxis];
            var mapedOutAxes = forwardDict[dimAxis];
            if (mapedOutAxes.Count == 1)
            {
                var outAxis = mapedOutAxes.First();
                var inAxes = backwardDict[outAxis];
                if (inAxes.Count == 1)
                {
                    ndsbp[outAxis] = inSBP;
                }
                else
                {
                    if (inAxes.All(i => forwardDict[i].Count == 1 && forwardDict[i].First() == outAxis))
                    {
                        var splitAxes = inAxes.Where(i => inType.AxisPolices[i] is SBPSplit);
                        if (splitAxes.Any())
                        {
                            ndsbp[outAxis] = new SBPSplit(splitAxes.ToArray());
                        }
                        else
                        {
                            ndsbp[outAxis] = SBP.B;
                        }
                    }
                }
            }
            else
            {
                if (mapedOutAxes.All(o => backwardDict[o].Count == 1))
                {
                    if (inSBP is SBPSplit split)
                    {
                        // TODO: may split to other axis
                        var firstValidAxis = mapedOutAxes.Where(axis => newShape[axis] > 1).First();
                        if (newShape[firstValidAxis] % split.Axes.Select(a => inType.Placement.Hierarchy[a]).Aggregate(1, (x, y) => x * y) == 0)
                        {
                            foreach (var oa in mapedOutAxes)
                            {
                                ndsbp[oa] = oa == firstValidAxis ? inSBP : SBP.B;
                            }
                        }
                        else
                        {
                            return invalidType;
                        }
                    }
                    else
                    {
                        foreach (var oa in mapedOutAxes)
                        {
                            ndsbp[oa] = SBP.B;
                        }
                    }
                }
                else
                {
                    return invalidType;
                }
            }

            // TODO: may add other complex reshape.
        }
#endif
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

        if (dataType.IsFloat() && dataType != DataTypes.Float32)
        {
            return Value.FromTensor(reshaped.ToTensor(tensorType).CastTo(dataType));
        }
        else
        {
            return Value.FromTensor(reshaped.ToTensor(tensorType));
        }
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

        return VisitDistributedType(inputType, outTensorType.Shape);
    }
}
