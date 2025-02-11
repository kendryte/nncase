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
public class ReshapeEvaluator : IEvaluator<Reshape>, ITypeInferencer<Reshape>, ICostEvaluator<Reshape>, IShapeEvaluator<Reshape>, IMetricEvaluator<Reshape>
{
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

    public Expr Visit(IShapeEvaluateContext context, Reshape target)
    {
        var inShape = context.GetArgumentShape(target, Reshape.Input);
        var shape = context.GetArgument(target, Reshape.Shape);
        return IR.F.ShapeExpr.ReshapeShape(inShape, shape);
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
        var shapeDims = new Shape(Enumerable.Range(0, rank).Select(i => shape[i]).ToArray());
        var outputShape = new Dimension[rank];

        // todo use egraph simplify.
        var minus1Dim = FixedAndDynamicDimension.Abs(input.Shape.ProdFixedAndDynamic() / shapeDims.ProdFixedAndDynamic());
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
                        outputShape[i] = Select(NotEqual(shapeDim.Value, -1L), shapeDim.Value, minus1Dim.ToExpr());
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

        var newShape = outTensorType.Shape;
        var oldShape = inputType.TensorType.Shape;

        // check is unsequeeze/sequeeze
        if (Enumerable.SequenceEqual(oldShape.Where(i => i != 1).ToArray(), newShape.Where(i => i != 1).ToArray()))
        {
            if (oldShape.Count < newShape.Count)
            {
                var axis = 0;
                var axisMap = new Dictionary<int, int>();
                if (!oldShape.IsScalar)
                {
                    for (var n = 0; n < newShape.Count; n++)
                    {
                        if (newShape[n] == oldShape[axis])
                        {
                            axisMap.Add(axis++, n);
                            if (axis >= oldShape.Count)
                            {
                                break;
                            }
                        }
                    }
                }

                var ndsbp = new SBP[inputType.Placement.Rank];
                for (int i = 0; i < inputType.Placement.Rank; i++)
                {
                    ndsbp[i] = inputType.NdSBP[i] switch
                    {
                        SBPSplit { Axis: int sx } => SBPSplit.S(axisMap[sx]),
                        SBP sbp => sbp,
                    };
                }

                return inputType with { TensorType = outTensorType, NdSBP = new(ndsbp) };
            }
            else if (oldShape.Count > newShape.Count)
            {
                var axis = 0;
                var axisMap = new Dictionary<int, int>();
                for (var o = 0; o < oldShape.Count; o++)
                {
                    if (axis < newShape.Count && oldShape[o] == newShape[axis])
                    {
                        axisMap.Add(o, axis++);
                        if (axis >= newShape.Count)
                        {
                            break;
                        }
                    }
                }

                var ndsbp = new SBP[inputType.Placement.Rank];
                for (int i = 0; i < inputType.Placement.Rank; i++)
                {
                    ndsbp[i] = inputType.NdSBP[i] switch
                    {
                        SBPSplit { Axis: int sx } => SBPSplit.S(axisMap[sx]),
                        SBP sbp => sbp,
                    };
                }

                return inputType with { TensorType = outTensorType, NdSBP = new(ndsbp) };
            }
        }

        // not the squeeze or unsqueeze
        if (!inputType.NdSBP.Any(sbp => sbp is SBPSplit))
        {
            return inputType with { TensorType = outTensorType, NdSBP = inputType.NdSBP };
        }

        return invalid;
    }
}
