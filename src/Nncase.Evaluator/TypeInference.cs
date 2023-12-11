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

        // If any input is not fixed, result is not fixed
        if (inputs.Any(x => !x.Shape.IsFixed))
        {
            // todo:
            // 1. multi same rank
            // 2. can broadcast rank -> biggest shape
            // 3. invalid rank
            var rank = inputs.OrderByDescending(x => x.Shape.Rank).First().Shape.Rank;
            return new TensorType(dataType, Shape.Unknown(rank));
        }

        var outputRank = inputs.Select(x => x.Shape.Rank).Max();
        var outputShape = new Dimension[outputRank];
        Span<int> inputDims = stackalloc int[inputs.Length];

        for (int dimIndex = 0; dimIndex < outputShape.Length; dimIndex++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var inShape = inputs[i].Shape;
                var inExtend = outputRank - inShape.Rank;
                var inDimIndex = dimIndex - inExtend;
                var inDim = inDimIndex < 0 ? 1 : inShape[inDimIndex].Value!.Value;
                if (inDim == 0)
                {
                    return new InvalidType("Input dimension should not be 0.");
                }

                inputDims[i] = inDim;
            }

            // 1. Sort descending
            inputDims.Sort((a, b) => b.CompareTo(a));

            // 2. Find first 1
            var firstOneIndex = inputDims.IndexOf(1);
            var expectedDim = inputDims[0];

            // 3. Dims before 1 are all same or 1 is not found, it's ok to broadcast
            if ((firstOneIndex == -1 && inputDims.AsValueEnumerable().Distinct().Count() == 1) ||
                ((firstOneIndex != -1) && inputDims[..firstOneIndex].AsValueEnumerable().All(x => x == expectedDim)))
            {
                outputShape[dimIndex] = expectedDim;
            }
            else
            {
                return new InvalidType("Inputs are not compatible to broadcast.");
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
        if (
            stride is TensorConst strideValue &&
            padding is TensorConst paddingValue &&
            dilation is TensorConst dilation_con &&
            groups is TensorConst groups_con &&
            input.Shape.IsFixed &&
            weights.Shape.IsFixed)
        {
            var ts_stride = strideValue.Value.Cast<int>();
            var ts_padding = paddingValue.Value.Cast<int>();
            var ts_dilation = dilation_con.Value.Cast<int>();
            var groups_v = groups_con.Value.ToScalar<int>();
            if (!(input.Shape[1].FixedValue >= groups_v && (input.Shape[1].FixedValue % groups_v) == 0))
            {
                return new InvalidType($"The Input Channel / Groups Error ({input.Shape[1].FixedValue}/{groups_v})");
            }

            if ((input.Shape[1] / groups_v) != weights.Shape[1])
            {
                return new InvalidType($"The input channel {input.Shape[1]} / {groups_v} != {weights.Shape[1]}");
            }

            outShape[2] = GetWindowedOutputSize(
                input.Shape[2].FixedValue + ts_padding[0, 0] + ts_padding[0, 1],
                weights.Shape[2].FixedValue,
                ts_stride[0],
                ts_dilation[0],
                false);
            outShape[3] = GetWindowedOutputSize(
                input.Shape[3].FixedValue + ts_padding[1, 0] + ts_padding[1, 1],
                weights.Shape[3].FixedValue,
                ts_stride[1],
                ts_dilation[1],
                false);
        }
        else
        {
            outShape[2] = outShape[3] = Dimension.Unknown;
        }

        return input with { Shape = new Shape(outShape) };
    }

    /// <summary>
    /// Pad Type Infer.
    /// </summary>
    public static IRType PadType(TensorType input, Expr pads, Expr pad)
    {
        if (input.Shape.IsUnranked)
        {
            return input;
        }

        if (pad.CheckedType is TensorType padValueType)
        {
            if (padValueType.DType != input.DType)
            {
                return new InvalidType($"Pad value and input must have same type, " +
                                       $"input:{input.DType}, padValue:{padValueType.DType}");
            }
        }

        if (pads is TensorConst paddings)
        {
            var tpads = paddings.Value.Cast<int>();
            var newShape = input.Shape.ToList();
            int channel = tpads.Dimensions[0];
            for (int i = 0; i < channel; i++)
            {
                newShape[newShape.Count - channel + i] += tpads[i, 0] + tpads[i, 1];
            }

            return new TensorType(input.DType, new Shape(newShape));
        }
        else
        {
            return new TensorType(input.DType, Shape.Unknown(input.Shape.Rank));
        }
    }

    /// <summary>
    /// ReduceWindow2D Type Infer.
    /// </summary>
    public static IRType ReduceWindow2DType(TensorType input, Expr filter, Expr stride, Expr padding, Expr ceilMode)
    {
        var outShape = input.Shape.ToList();
        if (
            filter is TensorConst filterValue &&
            stride is TensorConst strideValue &&
            padding is TensorConst paddingValue &&
            ceilMode is TensorConst ceilModeValue)
        {
            var ts_filter = filterValue.Value.Cast<int>();
            var ts_stride = strideValue.Value.Cast<int>();
            var ceilModeV = ceilModeValue.Value.ToScalar<bool>();
            var ts_padding = paddingValue.Value.Cast<int>();
            if (ts_padding.Rank != 2)
            {
                return new InvalidType($"The padding shape {ts_padding.Shape} is not support!");
            }

            var padh = ts_padding[0, 0] + ts_padding[0, 1];
            var padw = ts_padding[1, 0] + ts_padding[1, 1];
            outShape[2] = input.Shape[2].IsUnknown
                ? Dimension.Unknown
                : GetWindowedOutputSize(input.Shape[2].FixedValue + padh, ts_filter[0], ts_stride[0], 1, false, ceilModeV);
            outShape[3] = input.Shape[3].IsUnknown
                ? Dimension.Unknown
                : GetWindowedOutputSize(input.Shape[3].FixedValue + padw, ts_filter[1], ts_stride[1], 1, false, ceilModeV);

            return input with { Shape = new Shape(outShape) };
        }

        return input with { Shape = Shape.Unknown(4) };
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
            if (input.Shape.Count != permt.Length)
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
        var out_shape = input.Shape.ToArray();
        if (newSize is TensorConst new_size_con)
        {
            var ts_new_size = new_size_con.Value.ToArray<int>();
            switch (out_shape.Length)
            {
                case 2 or 3: // [h,w] ,[h,w,c]
                    out_shape[0] = ts_new_size[0];
                    out_shape[1] = ts_new_size[1];
                    break;
                case > 3: // resize [n,c,h,w]
                    out_shape[^2] = ts_new_size[^2]; // h
                    out_shape[^1] = ts_new_size[^1]; // w
                    break;
            }
        }
        else
        {
            switch (out_shape.Length)
            {
                case 2 or 3:
                    out_shape[0] = Dimension.Unknown;
                    out_shape[1] = Dimension.Unknown;
                    break;
                case > 3:
                    out_shape[^2] = Dimension.Unknown;
                    out_shape[^1] = Dimension.Unknown;
                    break;
            }
        }

        // for roi amount.
        if (inputbbox is not null && out_shape.Length == 4)
        {
            out_shape[0] = out_shape[0] * inputbbox.Shape[0].FixedValue;
        }

        return input with { Shape = new Shape(out_shape) };
    }

    /// <summary>
    /// input x is -1?.
    /// </summary>
    public static bool IsMinus1(int x) => x == -1;

    public static Shape ReshapeTo(TensorType tensorType)
    {
        var shape = tensorType.Shape;
        if (shape.IsRanked && shape[0].IsFixed)
        {
            Trace.Assert(shape.Count != 0);
            return Shape.Unknown(shape[0].FixedValue);
        }
        else
        {
            return Shape.Unranked;
        }
    }

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
            _ => new InvalidType($"Inputs of if should be same IRType Kind, but then:{thenType}, else: {elseType}"),
        };
    }
}
