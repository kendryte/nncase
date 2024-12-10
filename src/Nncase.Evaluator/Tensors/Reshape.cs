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
    public static IRType VisitDistributedType(DistributedType inType, int[] newShape)
    {
        var inShape = inType.TensorType.Shape.ToValueArray();
        IRType invalidType = new InvalidType($"not supported reshape {inType} to [{string.Join(",", newShape)}]");
        if (!IRUtility.TryGetShapeMapMatrix(inShape, newShape, out var mat))
        {
            return invalidType;
        }

        var (forwardDict, backwardDict) = IRUtility.ShapeMapMatrixAsDict(mat);
        var ndsbp = new SBP[inType.NdSBP.Count];
        for (int meshAxis = 0; meshAxis < inType.NdSBP.Count; meshAxis++)
        {
            var inSBP = inType.NdSBP[meshAxis];
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
                            var restSize = restAxes.Aggregate(1, (x, i) => x * newShape[i]);
                            if (restSize < (inShape[si.Axis] / inType.Placement.Hierarchy[meshAxis]))
                            {
                                ndsbp[meshAxis] = SBP.S(firstValidAxis);
                            } else {
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

        return new DistributedType(new TensorType(inType.TensorType.DType, newShape), ndsbp, inType.Placement);
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
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (context.GetArgument(target, Reshape.Shape) is TensorConst shapeConst)
        {
            var shapeValue = shapeConst.Value.ToArray<int>();
            var negCount = shapeValue.Count(IsMinus1);
            var shapeSize = shapeValue.Aggregate(1, (x, y) => x * y);
            if (negCount > 1)
            {
                return new InvalidType(
                    $"Reshape at most one dimension of the new shape can be -1," +
                    $" shape:{shapeValue}");
            }

            if (input.Shape.IsFixed)
            {
                var inputSize = input.Shape.Prod().FixedValue;
                if (negCount < 1)
                {
                    if (inputSize != shapeSize)
                    {
                        return new InvalidType("Reshape input shape size and param shape size must be same," +
                                               $" shape:{shapeValue.ToArray().Aggregate(string.Empty, (s, i) => s + i + " ")}, input shape${string.Join(",", input.Shape)}");
                    }

                    return input with { Shape = new Shape(shapeValue) };
                }
                else
                {
                    shapeSize = -shapeSize;
                    var negIndex = shapeValue.Select((dim, index) => (dim, index)).First(x => IsMinus1(x.dim)).index;
                    if (inputSize % shapeSize != 0)
                    {
                        return new InvalidType("Reshape input size must be divisible by shapeSize when has -1");
                    }

                    shapeValue[negIndex] = inputSize / shapeSize;
                    return input with { Shape = new Shape(shapeValue) };
                }
            }
            else
            {
                return input with
                {
                    Shape = new Shape(shapeValue.Select(x => x == -1 ? Dimension.Unknown : x).ToArray()),
                };
            }
        }

        var targetType = context.CheckArgumentType<TensorType>(target, Reshape.Shape);
        var outShape = ReshapeTo(targetType);
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

        var newShape = outTensorType.Shape.ToValueArray();
        return VisitDistributedType(inputType, newShape);
    }
}
