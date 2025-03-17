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
using static Nncase.Evaluator.TypeInference;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
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
                            var mapedOutAxes = forwardDict[si.Axis];

                            // when input axis is splited-by-reshape, we can direct obtain the tensor which splited-by-sbp on the first maped axis.
                            if (mapedOutAxes.Count > 1)
                            {
                                var firstValidAxis = mapedOutAxes.Where(axis => newShape[axis] > 1).First();
                                var restAxes = mapedOutAxes.Skip(mapedOutAxes.IndexOf(firstValidAxis) + 1).ToArray();
                                var restSize = restAxes.Aggregate(1L, (x, i) => x * newShape[i]);
                                if (restSize < (inShape[si.Axis] / inType.Placement.Hierarchy[meshAxis]))
                                {
                                    ndsbp[meshAxis] = SBP.S(firstValidAxis);
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
                                if (inAxes.Where(inAxis => inShape[inAxis] != 1).First() == si.Axis)
                                {
                                    ndsbp[meshAxis] = SBP.S(outAxis);
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

            return new DistributedType(new TensorType(inType.TensorType.DType, newSymbolShape), DistributedUtility.NDSBPToAxisPolices(ndsbp, newShape.Length), inType.Placement);
        }
        else
        {
            var inShape = inType.TensorType.Shape;

            // check is unsequeeze/sequeeze
            if (Enumerable.SequenceEqual(inShape.Where(i => i != 1).ToArray(), newSymbolShape.Where(i => i != 1).ToArray()))
            {
                if (inShape.Count < newSymbolShape.Count)
                {
                    var axis = 0;
                    var axisMap = new Dictionary<int, int>();
                    if (!inShape.IsScalar)
                    {
                        for (var n = 0; n < newSymbolShape.Count; n++)
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

                    var ndsbp = new SBP[inType.Placement.Rank];
                    for (int i = 0; i < inType.Placement.Rank; i++)
                    {
                        ndsbp[i] = inType.NdSBP[i] switch
                        {
                            SBPSplit { Axis: int sx } => SBPSplit.S(axisMap[sx]),
                            SBP sbp => sbp,
                        };
                    }

                    return inType with { TensorType = new TensorType(inType.TensorType.DType, newSymbolShape), NdSBP = new(ndsbp) };
                }
                else if (inShape.Count > newSymbolShape.Count)
                {
                    var axis = 0;
                    var axisMap = new Dictionary<int, int>();
                    for (var o = 0; o < inShape.Count; o++)
                    {
                        if (axis < newSymbolShape.Count && inShape[o] == newSymbolShape[axis])
                        {
                            axisMap.Add(o, axis++);
                            if (axis >= newSymbolShape.Count)
                            {
                                break;
                            }
                        }
                    }

                    var ndsbp = new SBP[inType.Placement.Rank];
                    for (int i = 0; i < inType.Placement.Rank; i++)
                    {
                        ndsbp[i] = inType.NdSBP[i] switch
                        {
                            SBPSplit { Axis: int sx } => SBPSplit.S(axisMap[sx]),
                            SBP sbp => sbp,
                        };
                    }

                    return inType with { TensorType = new TensorType(inType.TensorType.DType, newSymbolShape), NdSBP = new(ndsbp) };
                }
            }

            // not the squeeze or unsqueeze
            if (!inType.NdSBP.Any(sbp => sbp is SBPSplit))
            {
                return inType with { TensorType = new TensorType(inType.TensorType.DType, newSymbolShape), NdSBP = inType.NdSBP };
            }
        }

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
        var input = context.GetOrtArgumentValue(reshape, Reshape.Input);
        var shape = context.GetArgumentValueAsArray<long>(reshape, Reshape.Shape);
        if (context.CurrentCall.CheckedType is AnyType)
        {
            return Value.FromTensor(OrtKI.Reshape(input, shape, 0).ToTensor());
        }

        var tensorType = context.CurrentCall.CheckedTensorType;
        var allowzero = tensorType.Shape.Contains(0) ? 1L : 0L;
        if (tensorType.DType is VectorType vtype)
        {
            shape = shape.Concat(vtype.Lanes.Select(i => (long)i)).ToArray();
        }

        var reshaped = OrtKI.Reshape(input, shape, allowzero);

        return Value.FromTensor(reshaped.ToTensor(tensorType));
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
        var shape = context.GetDimensionArgument(target, Reshape.Shape);
        var shapeType = context.CheckArgumentTensorTypeOrBroadcast(target, Reshape.Shape);
        if (shapeType.Shape.IsUnranked || !shapeType.Shape[0].IsFixed)
        {
            return input with { Shape = Shape.Unranked };
        }

        var rank = (int)shapeType.Shape[0].FixedValue;
        var shapeDims = new Shape((from i in Enumerable.Range(0, rank)
                                   let dim = shape[i]
                                   select i < input.Shape.Rank ? Dimension.Select(dim, 0, input.Shape[i], dim) : dim).ToArray());
        var minus1DimCount = shapeDims.Count(x => x.IsFixed && x.FixedValue == -1);
        var outputShape = new Dimension[rank];

        if (minus1DimCount > 1)
        {
            return new InvalidType($"More than one -1 in the shape is not supported");
        }

        var minus1DimValue = FixedAndDynamicDimension.TryDivExactly(input.Shape.ProdFixedAndDynamic(), shapeDims.ProdFixedAndDynamic());
        if (!minus1DimValue.HasValue || (minus1DimValue.Value.Dynamic is null && minus1DimValue.Value.Fixed > 1))
        {
            return new InvalidType($"Cannot reshape {input.Shape} to {shapeDims}");
        }

        var minus1Dim = FixedAndDynamicDimension.Abs(minus1DimValue.Value);
        for (var i = 0; i < rank; i++)
        {
            var shapeDim = shapeDims[i];
            if (shapeDim.IsFixed)
            {
                outputShape[i] = shapeDim.FixedValue == -1 ? minus1Dim.ToDimension() : shapeDim;
            }
            else
            {
                switch (shapeDim)
                {
                    case Dimension { Value: Var }:
                        outputShape[i] = shapeDim;
                        break;
                    default:
                        outputShape[i] = Dimension.Select(shapeDim, -1L, minus1Dim.ToDimension(), shapeDim);
                        break;
                }
            }
        }

        return input with { Shape = outputShape };
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
