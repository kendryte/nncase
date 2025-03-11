// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.IR;
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
            DistributedType d when d.NdSBP.All(x => x is SBPBroadCast) => WrapperException(d.TensorType),
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
        if (inputs.Any(x => x.DType != dataType))
        {
            return new InvalidType(
                $"Inputs of broadcast must have same datatype: {string.Join(",", inputs.Select(x => x.DType.GetDisplayName()))}");
        }

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
                var inShape = inputs[i].Shape;
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
                    outputShape[dimIndex] = IR.F.Math.Max(non1Dims.Select(x => x.Value));
                }
            }
        }

        return new TensorType(dataType, new Shape(outputShape));
    }

    /// <summary>
    /// Conv2D Type Infer.
    /// </summary>
    public static IRType Conv2DType(TensorType input, TensorType weights, Expr stride, Expr padding, Expr dilation, Expr groups)
    {
        if (input.Shape.IsUnranked)
        {
            return input with { Shape = Shape.Unknown(4) };
        }

        var outShape = input.Shape.ToList();
        outShape[1] = weights.Shape[0];
        if (groups is TensorConst groups_con &&
            input.Shape.IsFixed &&
            weights.Shape.IsFixed)
        {
            var groups_v = groups_con.Value.ToScalar<int>();
            if (!(input.Shape[1].FixedValue >= groups_v && (input.Shape[1].FixedValue % groups_v) == 0))
            {
                return new InvalidType($"The Input Channel / Groups Error ({input.Shape[1].FixedValue}/{groups_v})");
            }

            if ((input.Shape[1] / groups_v) != weights.Shape[1])
            {
                return new InvalidType($"The input channel {input.Shape[1]} / {groups_v} != {weights.Shape[1]}");
            }
        }

        outShape[2] = GetWindowedOutputSize(
            input.Shape[2] + padding[0, 0] + padding[0, 1],
            weights.Shape[2],
            stride[0],
            dilation[0],
            false);
        outShape[3] = GetWindowedOutputSize(
            input.Shape[3] + padding[1, 0] + padding[1, 1],
            weights.Shape[3],
            stride[1],
            dilation[1],
            false);

        return input with { Shape = new Shape(outShape) };
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

    public static Expr GetPaddings(Shape inputShape, Shape weightsShape, Expr strides, Expr dilations, bool same, bool lower = false)
    {
        var padH = GetWindowedPadding(inputShape[2], weightsShape[2], strides[0], dilations[0], same, lower);
        var padW = GetWindowedPadding(inputShape[3], weightsShape[3], strides[1], dilations[1], same, lower);
        return Dimension.ConcatPadding(padH, padW);
    }

    public static Dimension[] GetWindowedPadding(Dimension inputSize, Dimension filter, Dimension stride, Dimension dilation, bool same, bool lower = false)
    {
        var outputSize = GetWindowedOutputSize(inputSize, filter, stride, dilation, same, false);
        return GetWindowedPaddingValue(inputSize, outputSize, filter, stride, dilation, lower);
    }

    public static Dimension[] GetWindowedPaddingValue(Dimension inputSize, Dimension outputSize, Dimension filter, Dimension stride, Dimension dilation, bool lower)
    {
        var effectiveFilterSize = ((filter - 1L) * dilation) + 1L;
        var padding = Dimension.Max(0L, ((outputSize - 1L) * stride) + effectiveFilterSize - inputSize);
        var before = padding / 2L;
        var after = padding - (padding / 2L);
        if (lower)
        {
            return [Dimension.Max(before, after), Dimension.Min(before, after)];
        }

        return [before, after];
    }

    /// <summary>
    /// Pad Type Infer.
    /// </summary>
    public static IRType PadType(TensorType input, TensorType padsType, Expr pads, Expr pad)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (padsType.Shape.IsUnranked || padsType.Shape.Rank != 2 || !padsType.Shape[0].IsFixed)
        {
            return new InvalidType($"The padding shape {padsType.Shape} is not support!");
        }

        if (pad.CheckedType is TensorType padValueType)
        {
            if (padValueType.DType != input.DType)
            {
                return new InvalidType($"Pad value and input must have same type, " +
                                       $"input:{input.DType}, padValue:{padValueType.DType}");
            }
        }

        var newShape = input.Shape.ToList();
        var channel = (int)padsType.Shape[0].FixedValue;
        for (int i = 0; i < channel; i++)
        {
            newShape[newShape.Count - channel + i] += pads[i, 0] + pads[i, 1];
        }

        return new TensorType(input.DType, new Shape(newShape));
    }

    /// <summary>
    /// ReduceWindow2D Type Infer.
    /// </summary>
    public static IRType ReduceWindow2DType(TensorType input, Expr filter, Expr stride, Expr padding, Expr ceilMode)
    {
        if (padding.CheckedShape.Rank != 2)
        {
            return new InvalidType($"The padding shape {padding.CheckedShape} is not support!");
        }

        var outShape = input.Shape.ToArray();
        if (ceilMode is TensorConst ceilModeValue)
        {
            var ceilModeV = ceilModeValue.Value.ToScalar<bool>();

            var padh = padding[0, 0] + padding[0, 1];
            var padw = padding[1, 0] + padding[1, 1];
            outShape[2] = GetWindowedOutputSize(input.Shape[2] + padh, filter[0], filter[0], 1L, false, ceilModeV);
            outShape[3] = GetWindowedOutputSize(input.Shape[3] + padw, filter[1], filter[1], 1L, false, ceilModeV);

            return input with { Shape = new Shape(outShape) };
        }

        throw new NotImplementedException("CeilMode is not constant");
    }

    /// <summary>
    /// Reduce Type Infer.
    /// </summary>
    public static IRType ReduceType(TensorType input, Expr keepDims, Expr axis)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (input.Shape.IsScalar)
        {
            return new InvalidType("Reduce input shape should not be scalar");
        }

        if (keepDims is TensorConst keepDimsV &&
            axis is TensorConst axisValue)
        {
            var axes = axisValue.Value.Cast<int>();
            var keepDimsValue = keepDimsV.Value.ToScalar<int>();
            var outShape = input.Shape.ToArray();
            foreach (var a in axes)
            {
                var ax = Util.PositiveIndex(a, input);
                if (keepDimsValue == 1)
                {
                    outShape[ax] = 1;
                }
                else
                {
                    outShape[ax] = 0;
                }
            }

            return input with { Shape = new Shape(outShape.Where(x => x != 0)) };
        }

        return input with { Shape = Shape.Unranked };
    }

    public static Shape ApplyPerm(Shape inShape, int[] perm)
    {
        var outShape = inShape.ToArray();
        foreach (var i in Enumerable.Range(0, inShape.Rank))
        {
            outShape[i] = inShape[perm[i]];
        }

        return outShape;
    }

    /// <summary>
    /// Pack Type Infer.
    /// </summary>
    public static IRType PackType(TensorType input, IRArray<int> lanes, IRArray<int> axes)
    {
        var vType = new VectorType(input.DType, lanes);
        if (input.Shape.IsRanked)
        {
            var dims = input.Shape.ToList();
            foreach (var (lane, axis) in lanes.Zip(axes))
            {
                if (dims[axis].IsFixed)
                {
                    dims[axis] = MathUtility.CeilDiv(dims[axis].FixedValue, lane);
                }
            }

            return new TensorType(vType, new Shape(dims));
        }

        return new TensorType(vType, Shape.Unranked);
    }

    public static IRType UnpackType(TensorType input, IRArray<int> axes)
    {
        if (input.DType is not VectorType vtype)
        {
            return new InvalidType("input.DType is not VectorType vtype");
        }

        if (input.Shape.IsRanked)
        {
            var dims = input.Shape.ToList();
            foreach (var (lanes, axis) in vtype.Lanes.Zip(axes))
            {
                dims[axis] *= lanes;
            }

            return new TensorType(vtype.ElemType, new Shape(dims));
        }

        return new TensorType(vtype.ElemType, Shape.Unranked);
    }

    /// <summary>
    /// Transpose Type Infer.
    /// </summary>
    public static IRType TransposeType(TensorType input, Expr perm)
    {
        if (perm is TensorConst permValue)
        {
            if (input.Shape.IsUnranked)
            {
                return input;
            }

            var permt = permValue.Value.ToArray<int>();
            if (input.Shape.Rank != permt.Length)
            {
                return new InvalidType("Transpose shoud perm.size == inShape.size");
            }

            var outShape = ApplyPerm(input.Shape, permt);
            return input with { Shape = outShape };
        }

        return input with { Shape = Shape.Unranked };
    }

    /// <summary>
    /// Resize Type Infer.
    /// </summary>
    public static IRType ResizeType(TensorType input, Expr newSize, TensorType? inputbbox)
    {
        var outShape = input.Shape.ToArray();
        switch (outShape.Length)
        {
            case 2 or 3: // [h,w] ,[h,w,c]
                outShape[0] = newSize[0];
                outShape[1] = newSize[1];
                break;
            case > 3: // resize [n,c,h,w]
                outShape[^2] = newSize[^2]; // h
                outShape[^1] = newSize[^1]; // w
                break;
        }

        // for roi amount.
        if (inputbbox is not null && outShape.Length == 4)
        {
            outShape[0] = outShape[0] * inputbbox.Shape[0].FixedValue;
        }

        return input with { Shape = new Shape(outShape) };
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
            var ndsbp = new IRArray<SBP>(Enumerable.Repeat(SBP.B, placement.Rank));
            foreach (ref var newType in newTypes.AsSpan())
            {
                if (newType is TensorType tensorType)
                {
                    newType = new DistributedType(tensorType, ndsbp, placement);
                }
            }

            return newTypes;
        }

        return types;
    }

    public static Shape ExpandShape(Shape inShape, Shape expandShape)
    {
        if (inShape.IsUnranked || expandShape.IsUnranked)
        {
            return Shape.Unranked;
        }

        var dimExtends = expandShape.Rank - inShape.Rank;
        var newDims = expandShape.ToArray();

        // dimsExtends may be negative
        for (int i = Math.Max(0, dimExtends); i < newDims.Length; i++)
        {
            var inDimIndex = i - dimExtends;
            ref var dimValue = ref newDims[i];
            dimValue = Dimension.Select(dimValue, 1L, inShape[inDimIndex], dimValue);
        }

        newDims = inShape.Take(dimExtends < 0 ? -dimExtends : 0).Concat(newDims).ToArray();
        return new Shape(newDims);
    }

    public static Shape ReshapeShape(Shape inShape, Expr newShape, TensorType? shapeType = null)
    {
        shapeType ??= (TensorType)newShape.CheckedType;

        if (shapeType.Shape.IsUnranked || !shapeType.Shape[0].IsFixed)
        {
            return Shape.Unranked;
        }

        var rank = (int)shapeType.Shape[0].FixedValue;
        var shapeDims = new Shape((from i in Enumerable.Range(0, rank)
                                   let dim = newShape[i]
                                   select i < inShape.Rank ? Dimension.Select(dim, 0, inShape[i], dim) : dim).ToArray());
        var minus1DimCount = shapeDims.Count(x => x.IsFixed && x.FixedValue == -1);
        var outputShape = new Dimension[rank];

        if (minus1DimCount > 1)
        {
            throw new TypeInferenceInterruptException(new InvalidType($"More than one -1 in the shape is not supported"));
        }

        var minus1DimValue = FixedAndDynamicDimension.TryDivExactly(inShape.ProdFixedAndDynamic(), shapeDims.ProdFixedAndDynamic());
        if (!minus1DimValue.HasValue || (minus1DimValue.Value.Dynamic is null && minus1DimValue.Value.Fixed > 1))
        {
            throw new TypeInferenceInterruptException(new InvalidType($"Cannot reshape {inShape} to {shapeDims}"));
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

        return outputShape;
    }
}
