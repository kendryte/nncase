// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.TIR;
using Nncase.Utilities;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.Evaluator;

/// <summary>
/// Type inference helper.
/// </summary>
public static class TypeInference
{
    /// <summary>
    /// Check argument type.
    /// </summary>
    /// <typeparam name="T">Desired type.</typeparam>
    /// <param name="context">Type inference context.</param>
    /// <param name="op">Operator.</param>
    /// <param name="parameter">Parameter.</param>
    /// <param name="reason">Reason text if not satisfied.</param>
    /// <returns>The desired type.</returns>
    public static T CheckArgumentType<T>(this ITypeInferenceContext context, Op op, ParameterInfo parameter, string? reason = null)
        where T : IRType
    {
        T WrapperException(T t)
        {
            try
            {
                return parameter.Pattern.Check(t, $"{op.GetType().Name}.{parameter.Name}");
            }
            catch (System.InvalidOperationException e)
            {
                throw new TypeInferenceInterruptException(new InvalidType(e.Message));
            }
        }

        return context.GetArgumentType(op, parameter) switch
        {
            T t => WrapperException(t),
            AnyType a => throw new TypeInferenceInterruptException(a),
            InvalidType iv => throw new TypeInferenceInterruptException(iv),
            var x => throw new TypeInferenceInterruptException(new InvalidType(reason ??
                                                                               $"{op.GetType().Name}.{parameter.Name} Must Be {typeof(T).Name} But Give {x.GetType().Name}.")),
        };
    }

    public static TensorType CheckArgumentTensorTypeOrBroadcast(this ITypeInferenceContext context, Op op, ParameterInfo parameter, string? reason = null)
    {
        TensorType WrapperException(TensorType t)
        {
            try
            {
                return parameter.Pattern.Check(t, $"{op.GetType().Name}.{parameter.Name}");
            }
            catch (System.InvalidOperationException e)
            {
                throw new TypeInferenceInterruptException(new InvalidType(e.Message));
            }
        }

        return context.GetArgumentType(op, parameter) switch
        {
            DistributedType d when d.AxisPolicies.All(x => x is SBPBroadCast) => WrapperException(d.TensorType),
            IRType t => CheckArgumentType<TensorType>(context, op, parameter, reason),
        };
    }

    /// <summary>
    /// Throw <seealso cref="TypeInferenceInterruptException"/> if type is <seealso cref="AnyType"/> or <seealso cref="InvalidType"/>.
    /// </summary>
    /// <typeparam name="T">Type.</typeparam>
    /// <param name="t">Type instance.</param>
    /// <returns>Original type instance.</returns>
    public static T ThrowIfTypeInferenceInterrupt<T>(this T t)
        where T : IRType
    {
        return t switch
        {
            AnyType a => throw new TypeInferenceInterruptException(a),
            InvalidType i => throw new TypeInferenceInterruptException(i),
            T other => other,
        };
    }

    /// <summary>
    /// Broadcast input shapes.
    /// </summary>
    /// <param name="inputs">Input shapes.</param>
    /// <returns>Broadcasted shape.</returns>
    public static IRType BroadcastType(params TensorType[] inputs)
    {
        if (inputs.Length < 2)
        {
            throw new ArgumentException("Broadcast must have 2 inputs at least.");
        }

        var dataType = inputs[0].DType;
        dataType = dataType is VectorType vt ? vt.ElemType : dataType;
        if (!inputs.All(x => x.DType == dataType || (x.DType is VectorType vt && vt.ElemType == dataType)))
        {
            return new InvalidType(
                $"Inputs of broadcast must have same datatype: {string.Join(",", inputs.Select(x => x.DType.GetDisplayName()))}");
        }

        dataType = inputs.Select(x => x.DType).OfType<VectorType>().FirstOrDefault() ?? dataType;
        return BroadcastType(dataType, inputs);
    }

    public static IRType BroadcastType(DataType dataType, params TensorType[] inputs)
    {
        // If any input is invalid, result is invalid
        if (inputs.Any(x => x.Shape.IsInvalid))
        {
            return TensorType.Invalid(dataType);
        }

        // If any input is not unranked, result is unranked
        if (inputs.Any(x => x.Shape.IsUnranked))
        {
            return TensorType.Unranked(dataType);
        }

        var outputRank = inputs.Select(x => x.Shape.Rank).Max();
        var outputShape = new Dimension[outputRank];
        var inputDims = new Dimension[inputs.Length];

        for (int dimIndex = 0; dimIndex < outputShape.Length; dimIndex++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var inShape = (RankedShape)inputs[i].Shape;
                var inExtend = outputRank - inShape.Rank;
                var inDimIndex = dimIndex - inExtend;
                var inDim = inDimIndex < 0 ? 1 : inShape[inDimIndex];
                if (inDim is Dimension { IsFixed: true, FixedValue: 0 })
                {
                    return new InvalidType("Input dimension should not be 0.");
                }

                inputDims[i] = inDim;
            }

            var non1Dims = inputDims.Where(x => x.IsDynamic || x.IsUnknown || x.FixedValue != 1).ToHashSet();
            if (non1Dims.Count == 0)
            {
                outputShape[dimIndex] = 1;
            }
            else
            {
                var expectedDim = non1Dims.First();
                if (non1Dims.Count == 1)
                {
                    outputShape[dimIndex] = expectedDim;
                }
                else if (non1Dims.Any(x => x.IsFixed))
                {
                    var fixedDim = non1Dims.First(x => x.IsFixed).FixedValue;
                    if (non1Dims.Any(x => x.IsFixed && x.FixedValue != fixedDim))
                    {
                        return new InvalidType("Inputs are not compatible to broadcast.");
                    }
                    else
                    {
                        outputShape[dimIndex] = fixedDim;
                    }
                }
                else
                {
                    outputShape[dimIndex] = Dimension.Max(non1Dims.ToArray());
                }
            }
        }

        return new TensorType(dataType, new RankedShape(outputShape));
    }

    /// <summary>
    /// Broadcast input shapes.
    /// </summary>
    /// <param name="cond">Condition shape.</param>
    /// <param name="inputs">Input shapes.</param>
    /// <returns>Broadcasted shape.</returns>
    public static IRType WhereType(TensorType cond, params TensorType[] inputs)
    {
        if (inputs.Length < 2)
        {
            throw new ArgumentException("Broadcast must have 2 inputs at least.");
        }

        var dataType = inputs[0].DType;
        dataType = dataType is VectorType vt ? vt.ElemType : dataType;
        if (!inputs.All(x => x.DType == dataType || (x.DType is VectorType vt && vt.ElemType == dataType)))
        {
            return new InvalidType(
                $"Inputs of broadcast must have same datatype: {string.Join(",", inputs.Select(x => x.DType.GetDisplayName()))}");
        }

        dataType = inputs.Select(x => x.DType).OfType<VectorType>().FirstOrDefault() ?? dataType;
        if (cond.DType is MaskVectorType maskVectorType)
        {
            if (dataType is VectorType vt2)
            {
                if (vt2.Lanes.Count != 1 || vt2.Lanes[0] != maskVectorType.Lanes)
                {
                    return new InvalidType(
                            $"The cond mask vector lanes {maskVectorType.Lanes} is not compatible with input vector lanes {vt2.Lanes}");
                }
            }
            else
            {
                dataType = new VectorType(dataType, maskVectorType.Lanes);
            }
        }

        return BroadcastType(dataType, [cond, .. inputs]);
    }

    /// <summary>
    /// Conv2D Type Infer.
    /// </summary>
    public static IRType Conv2DType(TensorType input, TensorType weights, Shape strides, Paddings paddings, Shape dilations, Dimension groups)
    {
        if (input.Shape is RankedShape inShape)
        {
            var outShape = inShape.ToList();
            outShape[1] = weights.Shape[0];
            if (groups.IsFixed &&
                input.Shape.IsFixed &&
                weights.Shape.IsFixed)
            {
                var groups_v = groups.FixedValue;
                if (!(input.Shape[1].FixedValue >= groups_v && (input.Shape[1].FixedValue % groups_v) == 0))
                {
                    return new InvalidType($"The Input Channel / Groups Error ({input.Shape[1].FixedValue}/{groups_v})");
                }

                if ((input.Shape[1] / groups_v) != weights.Shape[1])
                {
                    return new InvalidType($"The input channel {input.Shape[1]} / {groups_v} != {weights.Shape[1]}");
                }
            }

            for (int i = 0; i < 2; i++)
            {
                outShape[i + 2] = GetWindowedOutputSize(
                    input.Shape[i + 2] + paddings[i].Before + paddings[i].After,
                    weights.Shape[i + 2],
                    strides[i],
                    dilations[i],
                    false);
            }

            return input with { Shape = new RankedShape(outShape) };
        }

        return input with { Shape = Shape.Unknown(4) };
    }

    /// <summary>
    /// get padding windows output size.
    /// </summary>
    public static Dimension GetWindowedOutputSize(Dimension size, Dimension filter, Dimension stride, Dimension dilation, bool same, bool ceilMode = false)
    {
        var effective_filter_size = ((filter - 1L) * dilation) + 1L;
        if (same)
        {
            return (size + stride - 1L) / stride;
        }
        else
        {
            if (!ceilMode)
            {
                return (size - effective_filter_size + stride) / stride;
            }
            else
            {
                return Dimension.CeilDiv(size - effective_filter_size + stride, stride);
            }
        }
    }

    /// <summary>
    /// GetWindowedOutputSize.
    /// </summary>
    public static int GetWindowedOutputSize(int size, int filter, int stride, int dilation, (int Before, int After) padding)
    {
        var effective_filter_size = ((filter - 1) * dilation) + 1;
        return (size + padding.Before + padding.After - effective_filter_size + stride) / stride;
    }

    public static Paddings GetPaddings(Shape inputShape, Shape weightsShape, Shape strides, Shape dilations, bool same, bool lower = false)
    {
        var padH = GetWindowedPadding(inputShape[2], weightsShape[2], strides[0], dilations[0], same, lower);
        var padW = GetWindowedPadding(inputShape[3], weightsShape[3], strides[1], dilations[1], same, lower);
        return new[] { padH, padW };
    }

    public static Padding GetWindowedPadding(Dimension inputSize, Dimension filter, Dimension stride, Dimension dilation, bool same, bool lower = false)
    {
        var outputSize = GetWindowedOutputSize(inputSize, filter, stride, dilation, same, false);
        return GetWindowedPaddingValue(inputSize, outputSize, filter, stride, dilation, lower);
    }

    public static Padding GetWindowedPaddingValue(Dimension inputSize, Dimension outputSize, Dimension filter, Dimension stride, Dimension dilation, bool lower)
    {
        var effectiveFilterSize = ((filter - 1L) * dilation) + 1L;
        var padding = Dimension.Max(0L, ((outputSize - 1L) * stride) + effectiveFilterSize - inputSize);
        var before = padding / 2L;
        var after = padding - (padding / 2L);
        if (lower)
        {
            return (Dimension.Max(before, after), Dimension.Min(before, after));
        }

        return (before, after);
    }

    /// <summary>
    /// Pad Type Infer.
    /// </summary>
    public static IRType PadType(TensorType input, Paddings pads, Expr padValue)
    {
        if (input.Shape is not RankedShape inShape)
        {
            return input;
        }

        if (padValue.CheckedType is TensorType padValueType)
        {
            if (!(padValueType.DType == input.DType
                || (input.DType is VectorType vt && padValueType.DType == vt.ElemType)))
            {
                return new InvalidType($"Pad value and input must have same type, " +
                                       $"input:{input.DType}, padValue:{padValueType.DType}");
            }
        }

        var newShape = inShape.ToList();
        var channel = pads.Rank;
        if (channel > newShape.Count)
        {
            return new InvalidType($"Pads rank {channel} is greater than input rank {newShape.Count}");
        }

        for (int i = 0; i < channel; i++)
        {
            newShape[newShape.Count - channel + i] += pads[i].Sum();
        }

        return new TensorType(input.DType, new RankedShape(newShape));
    }

    /// <summary>
    /// ReduceWindow2D Type Infer.
    /// </summary>
    public static IRType ReduceWindow2DType(TensorType input, Shape filters, Shape strides, Paddings paddings, Expr ceilMode)
    {
        if (paddings.Rank != 2)
        {
            return new InvalidType($"The padding shape {paddings.Rank} is not support!");
        }

        if (input.Shape is not RankedShape inShape)
        {
            return input;
        }

        var outShape = inShape.ToArray();
        if (ceilMode is TensorConst ceilModeValue)
        {
            var ceilModeV = ceilModeValue.Value.ToScalar<bool>();

            var padh = paddings[0].Before + paddings[0].After;
            var padw = paddings[1].Before + paddings[1].After;
            outShape[2] = GetWindowedOutputSize(inShape[2] + padh, filters[0], filters[0], 1L, false, ceilModeV);
            outShape[3] = GetWindowedOutputSize(inShape[3] + padw, filters[1], filters[1], 1L, false, ceilModeV);

            return input with { Shape = new RankedShape(outShape) };
        }

        throw new NotImplementedException("CeilMode is not constant");
    }

    /// <summary>
    /// Reduce Type Infer.
    /// </summary>
    public static IRType ReduceType(TensorType input, Expr keepDims, Shape axes)
    {
        if (input.Shape is not RankedShape inShape)
        {
            return input;
        }

        if (input.Shape.IsScalar)
        {
            return new InvalidType("Reduce input shape should not be scalar");
        }

        if (keepDims is TensorConst keepDimsV &&
            axes.IsFixed)
        {
            var keepDimsValue = keepDimsV.Value.ToScalar<int>();
            var outShape = inShape.ToArray();
            foreach (var a in (RankedShape)axes)
            {
                var ax = Dimension.Positive(a, input.Shape.Rank).FixedValue;
                if (keepDimsValue == 1)
                {
                    outShape[ax] = 1;
                }
                else
                {
                    outShape[ax] = 0;
                }
            }

            return input with { Shape = new RankedShape(outShape.Where(x => x != 0)) };
        }

        return input with { Shape = Shape.Unranked };
    }

    public static RankedShape ApplyPerm(RankedShape inShape, Shape perm)
    {
        var outShape = inShape.ToArray();
        foreach (var i in Enumerable.Range(0, inShape.Rank))
        {
            outShape[i] = inShape[perm[i].FixedValue];
        }

        return outShape;
    }

    /// <summary>
    /// Vectorize Type Infer.
    /// </summary>
    public static IRType VectorizeType(TensorType input, ReadOnlySpan<int> lanes, ReadOnlySpan<int> axes)
    {
        if (input.DType is BooleanType)
        {
            return new InvalidType("VectorizeType does not support BooleanType input");
        }

        var vType = new VectorType(input.DType, lanes.ToArray());
        return VectorizeType(input.Shape, vType, lanes, axes);
    }

    public static IRType VectorizeMaskType(TensorType input, MaskVectorStyle style, int elementBits, int lanes, int axis)
    {
        if (input.DType is not BooleanType)
        {
            return new InvalidType("input.DType is not BooleanType");
        }

        var vType = new MaskVectorType(style, elementBits, lanes);
        return VectorizeType(input.Shape, vType, [lanes], [axis]);
    }

    public static IRType DevectorizeType(TensorType input, IRArray<int> axes)
    {
        return input.DType switch
        {
            MaskVectorType vt => DevectorizeType(input.Shape, DataTypes.Boolean, [vt.Lanes], axes),
            VectorType vt => DevectorizeType(input.Shape, vt.ElemType, vt.Lanes, axes),
            _ => new InvalidType($"DevectorizeType does not support {input.DType}"),
        };
    }

    /// <summary>
    /// Transpose Type Infer.
    /// </summary>
    public static IRType TransposeType(TensorType input, Shape perm)
    {
        if (input.Shape is not RankedShape inShape)
        {
            return input;
        }

        if (!perm.IsFixed)
        {
            return input with { Shape = Shape.Unranked };
        }

        if (inShape.Rank != perm.Rank)
        {
            return new InvalidType("Transpose shoud perm.size == inShape.size");
        }

        var outShape = ApplyPerm(inShape, perm);
        return input with { Shape = outShape };
    }

    /// <summary>
    /// Resize Type Infer.
    /// </summary>
    public static IRType ResizeType(TensorType input, Shape newSize, TensorType? inputbbox)
    {
        if (input.Shape is not RankedShape inShape)
        {
            return input;
        }

        if (newSize is not RankedShape rankedNewSize)
        {
            return input with { Shape = Shape.Unranked };
        }

        var outShape = inShape.ToArray();
        switch (outShape.Length)
        {
            case 2 or 3: // [h,w] ,[h,w,c]
                outShape[0] = rankedNewSize[0];
                outShape[1] = rankedNewSize[1];
                break;
            case > 3: // resize [n,c,h,w]
                outShape[^2] = rankedNewSize[^2]; // h
                outShape[^1] = rankedNewSize[^1]; // w
                break;
        }

        // for roi amount.
        if (inputbbox is not null && outShape.Length == 4)
        {
            outShape[0] = outShape[0] * inputbbox.Shape[0].FixedValue;
        }

        return input with { Shape = new RankedShape(outShape) };
    }

    /// <summary>
    /// input x is -1?.
    /// </summary>
    public static bool IsMinus1(long x) => x == -1;

    /// <summary>
    /// Infer CommonType for inputs.
    /// </summary>
    /// <param name="thenType">Then type.</param>
    /// <param name="elseType">Else type.</param>
    /// <returns>IRType.</returns>
    public static IRType CommonType(IRType thenType, IRType elseType)
    {
        IRType CommonTypeImpl(TensorType a, TensorType b)
        {
            if (a == b)
            {
                return a;
            }

            if (a.DType != b.DType)
            {
                return new InvalidType($"Inputs DType of if should be same, then: {a.DType}, else: {b.DType}");
            }

            if (a.Shape.IsUnranked || b.Shape.IsUnranked || a.Shape.Rank != b.Shape.Rank)
            {
                return new TensorType(a.DType, Shape.Unranked);
            }

            return new TensorType(a.DType, Shape.Unknown(a.Shape.Rank));
        }

        IRType DistributedCommonTypeImpl(DistributedType a, DistributedType b)
        {
            var tA = DistributedUtility.GetDividedTensorType(a);
            var tB = DistributedUtility.GetDividedTensorType(b);
            if (tA == tB)
            {
                return a;
            }

            if (tA.DType != tB.DType)
            {
                return new InvalidType($"Inputs DType of if should be same, then: {tA.DType}, else: {tB.DType}");
            }

            return new TensorType(tA.DType, Shape.Unknown(tA.Shape.Rank));
        }

        return (thenType, elseType) switch
        {
            (TensorType then, TensorType @else) => CommonTypeImpl(then, @else),
            (TupleType then, TupleType @else) => then.Count != @else.Count
                ? new InvalidType($"tuple Inputs of if should be same count, then: {then.Count}, else: {@else.Count}")
                : new TupleType(then.Zip(@else).Select(tuple => CommonType(tuple.First, tuple.Second))),
            (DistributedType then, DistributedType @else) => DistributedCommonTypeImpl(then, @else),
            (InvalidType then, _) => then,
            (_, InvalidType @else) => @else,
            _ => AnyType.Default,
        };
    }

    public static IRType[] BroadcastDistributeTypes(params IRType[] types)
    {
        var placement = types.OfType<DistributedType>().FirstOrDefault()?.Placement;
        if (placement != null)
        {
            var newTypes = types.ToArray();
            foreach (ref var newType in newTypes.AsSpan())
            {
                if (newType is TensorType tensorType && tensorType.Shape is RankedShape { Rank: var rank })
                {
                    var ndsbp = new IRArray<SBP>(Enumerable.Repeat(SBP.B, rank).ToImmutableArray<SBP>());
                    newType = new DistributedType(tensorType, ndsbp, placement);
                }
            }

            return newTypes;
        }

        return types;
    }

    public static Shape ExpandShape(Shape inShape, Shape expandShape)
    {
        if (inShape is not RankedShape inRankedShape
            || expandShape is not RankedShape expandRankedShape)
        {
            return Shape.Unranked;
        }

        var dimExtends = expandRankedShape.Rank - inRankedShape.Rank;
        var newDims = expandRankedShape.ToArray();

        // dimsExtends may be negative
        for (int i = Math.Max(0, dimExtends); i < newDims.Length; i++)
        {
            var inDimIndex = i - dimExtends;
            ref var dimValue = ref newDims[i];
            dimValue = Dimension.Select(dimValue, 1L, inRankedShape[inDimIndex], dimValue);
        }

        newDims = inRankedShape.Take(dimExtends < 0 ? -dimExtends : 0).Concat(newDims).ToArray();
        return new RankedShape(newDims);
    }

    public static Shape ReshapeShape(Shape inShape, Shape newShape)
    {
        if (inShape is not RankedShape inRankedShape
            || newShape is not RankedShape newRankedShape)
        {
            return Shape.Unranked;
        }

        var rank = newShape.Rank;
        var shapeDims = new RankedShape((from i in Enumerable.Range(0, rank)
                                         let dim = newRankedShape[i]
                                         select i < inRankedShape.Rank ? Dimension.Select(dim, 0, inRankedShape[i], dim) : dim).ToArray());
        var minus1DimCount = shapeDims.Count(x => x.IsFixed && x.FixedValue == -1);
        var outputShape = new Dimension[rank];

        if (minus1DimCount > 1)
        {
            throw new TypeInferenceInterruptException(new InvalidType($"More than one -1 in the shape is not supported"));
        }

        var remainder = inRankedShape.Prod() % shapeDims.Prod();
        var minus1DimValue = inRankedShape.Prod() / shapeDims.Prod();

        // Notes: when the remainder is not fixed, we cannot make sure the reshape is invalid
        if (remainder is DimConst { Value: not 0 }
            || minus1DimValue is DimConst { Value: > 1 })
        {
            throw new TypeInferenceInterruptException(new InvalidType($"Cannot reshape {inShape} to {shapeDims}"));
        }

        var minus1Dim = Dimension.Abs(minus1DimValue);
        for (var i = 0; i < rank; i++)
        {
            var shapeDim = shapeDims[i];
            outputShape[i] = Dimension.Select(shapeDim, -1L, minus1Dim, shapeDim);
        }

        return outputShape;
    }

    private static IRType VectorizeType(Shape inputShape, DataType vectorType, ReadOnlySpan<int> lanes, ReadOnlySpan<int> axes)
    {
        if (inputShape is RankedShape inShape)
        {
            var dims = inShape.ToList();
            for (int i = 0; i < axes.Length; i++)
            {
                var axis = axes[i];
                var lane = lanes[i];
                if (dims[axis].IsFixed)
                {
                    dims[axis] = MathUtility.CeilDiv(dims[axis].FixedValue, lane);
                }
                else
                {
                    dims[axis] = dims[axis] / lane;
                }
            }

            return new TensorType(vectorType, new RankedShape(dims));
        }

        return new TensorType(vectorType, Shape.Unranked);
    }

    private static IRType DevectorizeType(Shape inputShape, DataType elementType, ReadOnlySpan<int> lanes, ReadOnlySpan<int> axes)
    {
        if (inputShape is RankedShape inShape)
        {
            var dims = inShape.ToList();
            for (int i = 0; i < axes.Length; i++)
            {
                var axis = axes[i];
                var lane = lanes[i];
                dims[axis] *= lane;
            }

            return new TensorType(elementType, new RankedShape(dims));
        }

        return new TensorType(elementType, Shape.Unranked);
    }
}
