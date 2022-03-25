// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommonServiceLocator;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.IR;
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
            var x => throw new TypeInferenceInterruptException(new InvalidType(reason ?? $"{op.GetType().Name}.{parameter.Name} Must Be {typeof(T).Name} But Give {x.GetType().Name}.")),
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
    /// get pad value in axis
    /// </summary>
    /// <param name="pads"></param>
    /// <param name="axis"></param>
    /// <returns>(pad_before, pad_after)</returns>
    private static (int, int) GetPadByAxis(int[] pads, int axis)
    {
        return (pads[axis], pads[axis + pads.Length / 2]);
    }

    private static int GetPadHSum(int[] pads)
    {
        var axis = pads.Length / 2 - 2;
        return GetPadSumByAxis(pads, axis);
    }

    private static int GetPadWSum(int[] pads)
    {
        var axis = pads.Length / 2 - 1;
        return GetPadSumByAxis(pads, axis);
    }

    /// <summary>
    /// get pad sum in axis
    /// </summary>
    /// <param name="pads"></param>
    /// <param name="axis"></param>
    /// <returns>value = pad_before + pad_after</returns>
    private static int GetPadSumByAxis(int[] pads, int axis)
    {
        var paddings = GetPadByAxis(pads, axis);
        return paddings.Item1 + paddings.Item2;
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
            return new InvalidType("Inputs of broadcast must have same datatype.");
        }

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
            return inputs.OrderByDescending(x => x.Shape.Rank).First();
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
                    throw new InvalidOperationException("Input dimension should not be 0.");
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
        var outShape = input.Shape.ToList();
        outShape[1] = weights.Shape[0];
        if (
            stride is TensorConst strideValue &&
            padding is TensorConst paddingValue &&
            dilation is TensorConst dilation_con &&
            groups is TensorConst groups_con &&
            input.Shape[2].IsFixed &&
            input.Shape[3].IsFixed &&
            weights.Shape[2].IsFixed &&
            weights.Shape[3].IsFixed)
        {
            var ts_stride = strideValue.Value.Cast<int>();
            var ts_padding = paddingValue.Value.ToArray<int>();
            var ts_dilation = dilation_con.Value.Cast<int>();
            var groups_v = groups_con.Value.ToScalar<int>();

            outShape[2] = GetWindowedOutputSize(input.Shape[2].FixedValue + GetPadHSum(ts_padding),
                weights.Shape[2].FixedValue, ts_stride[0], ts_dilation[0], false);
            outShape[3] = GetWindowedOutputSize(input.Shape[3].FixedValue + GetPadWSum(ts_padding),
                weights.Shape[3].FixedValue, ts_stride[1], ts_dilation[1], false);
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
            var tpads = paddings.Value.ToArray<int>();
            var newShape = input.Shape.ToList();
            int channel = tpads.Length / 2;
            for (int i = 0; i < channel; i++)
            {
                newShape[newShape.Count - channel + i] += GetPadSumByAxis(tpads, i);
            }

            return new TensorType(input.DType, new Shape(newShape));
        }
        else
        {
            return AnyType.Default;
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
            ceilMode is TensorConst ceilModeValue
        )
        {
            var ts_filter = filterValue.Value.Cast<int>();
            var ts_stride = strideValue.Value.Cast<int>();
            var ceilModeV = ceilModeValue.Value.ToScalar<bool>();
            var ts_padding = paddingValue.Value.ToArray<int>();
            outShape[2] = input.Shape[2].IsUnknown
                ? Dimension.Unknown
                : GetWindowedOutputSize(input.Shape[2].FixedValue + GetPadHSum(ts_padding), ts_filter[0], ts_stride[0], 1, false, ceilModeV);
            outShape[3] = input.Shape[3].IsUnknown
                ? Dimension.Unknown
                : GetWindowedOutputSize(input.Shape[3].FixedValue + GetPadWSum(ts_padding), ts_filter[1], ts_stride[1], 1, false, ceilModeV);

            return input with { Shape = new Shape(outShape) };
        }

        return new InvalidType("Can't Infer Shape With Dynamic Input!");
    }

    /// <summary>
    /// Reduce Type Infer.
    /// </summary>
    public static IRType ReduceType(TensorType input, Expr keepDims, Expr axis)
    {
        if (keepDims is TensorConst keepDimsValue &&
            axis is TensorConst axisValue)
        {
            var axes = axisValue.Value.Cast<int>();
            var outShape = input.Shape.ToValueArray();
            foreach (var a in axes)
            {
                var ax = Util.PositiveIndex(a, input);
                if (keepDimsValue.Value.ToScalar<int>() == 1)
                {
                    outShape[ax] = 1;
                }
                else
                {

                    // todo: test
                    outShape[ax] = 0;
                }
            }

            return input with { Shape = new Shape(outShape.Where(x => x != -1)) };
        }

        return new InvalidType("Can't Infer Shape With Dynamic Input!");
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
                return new InvalidType("Transpose input should not be Unranked");
            }

            var permt = permValue.Value.Cast<int>();
            var inShape = input.Shape;
            var outShape = inShape.ToArray();
            foreach (var i in Enumerable.Range(0, inShape.Rank))
            {
                outShape[i] = inShape[permt[i]];
            }

            return input with { Shape = outShape };
        }

        return input with { Shape = new Shape(Enumerable.Repeat(Dimension.Unknown, input.Shape.Rank)) };
    }

    /// <summary>
    /// Resize Type Infer.
    /// </summary>
    public static IRType ResizeType(TensorType input, Expr newSize)
    {
        var out_shape = input.Shape.ToArray();
        if (newSize is TensorConst new_size_con)
        {
            var ts_new_size = new_size_con.Value.Cast<int>();
            switch (out_shape.Length)
            {
                case 2 or 3:
                    out_shape[0] = ts_new_size[0];
                    out_shape[1] = ts_new_size[1];
                    break;
                case > 3:
                    out_shape[^3] = ts_new_size[0];
                    out_shape[^2] = ts_new_size[1];
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
                    out_shape[^3] = Dimension.Unknown;
                    out_shape[^2] = Dimension.Unknown;
                    break;
            }
        }

        return input with { Shape = new Shape(out_shape) };
    }

    /// <summary>
    /// input x is -1?
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    public static bool IsMinus1(int x) => x == -1;
}
