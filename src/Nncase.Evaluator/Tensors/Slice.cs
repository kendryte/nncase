// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using DryIoc.FastExpressionCompiler.LightExpression;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Evaluator.Math;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using Tensorflow;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using Dimension = Nncase.IR.Dimension;
using Shape = Nncase.IR.Shape;
using Slice = Nncase.IR.Tensors.Slice;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Slice"/>.
/// </summary>
public class SliceEvaluator : IEvaluator<Slice>, ITypeInferencer<Slice>, ICostEvaluator<Slice>, IShapeEvaluator<Slice>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Slice sl)
    {
        var input = context.GetOrtArgumentValue(sl, Slice.Input);
        var begins = context.GetInt64OrtTensorArgumentValue(sl, Slice.Begins);
        var ends = context.GetInt64OrtTensorArgumentValue(sl, Slice.Ends);
        var axes = context.GetInt64OrtTensorArgumentValue(sl, Slice.Axes);
        var strides = context.GetInt64OrtTensorArgumentValue(sl, Slice.Strides);
        return OrtKI.Slice(input, begins, ends, axes, strides).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Slice target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Slice.Input);
        context.CheckArgumentType<TensorType>(target, Slice.Begins);
        context.CheckArgumentType<TensorType>(target, Slice.Ends);
        context.CheckArgumentType<TensorType>(target, Slice.Axes);
        context.CheckArgumentType<TensorType>(target, Slice.Strides);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Slice target)
    {
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Slice target)
    {
        var inShape = context.GetArgumentShape(target, Slice.Input);
        var begins = context.GetArgument(target, Slice.Begins);
        var ends = context.GetArgument(target, Slice.Ends);
        var strides = context.GetArgument(target, Slice.Strides);
        var axes = context.GetArgument(target, Slice.Axes);
        var size = context.GetArgument(target, Slice.Input).CheckedShape.Rank;

        Expr Translate(Expr x, Expr dim) => new If(x < 0, dim + x, Clamp(x, 0, dim));

        var axesValue = ((TensorConst)axes).Value.ToArray<int>();
        int j = 0;
        var outDims = Enumerable.Range(0, size).Select(i =>
        {
            var dim = Cast(inShape[i], DataTypes.Int32);
            if (!axesValue.Contains(i))
            {
                return dim;
            }

            Expr begin = Cast(Clamp(begins[j], (long)int.MinValue, (long)int.MaxValue), DataTypes.Int32);
            Expr end = Cast(Clamp(ends[j], (long)int.MinValue, (long)int.MaxValue), DataTypes.Int32);
            var stride = Cast(Clamp(strides[j], (long)int.MinValue, (long)int.MaxValue), DataTypes.Int32);
            var strideIsNeg = stride < 0;
            begin = new If(
                strideIsNeg,
                Clamp(begin, 0, dim - 1), Translate(begin, dim));
            end = new If(strideIsNeg, Clamp(end, -1, dim), Translate(end, dim));
            j++;
            return Ceil(Abs(end - begin) / Abs(stride));
        }).ToArray();

        return Stack(new IR.Tuple(outDims), 0);
    }

    /// <param name="axisConst">Axis.</param>
    /// <param name="input">Input type.</param>
    /// <param name="f">(index in axis, axis, inDim) -> outDim.</param>
    private Shape ApplyAxis(TensorConst axisConst, TensorType input, Func<int, int, int, Dimension> f)
    {
        if (input.Shape.IsUnranked)
        {
            return Shape.Unranked;
        }

        var outShape = input.Shape.ToArray();
        var axesTensor = axisConst.Value.Cast<int>();
        for (int i = 0; i < axesTensor.Length; i++)
        {
            var axisV = axesTensor[i];
            var axis = axisV < 0
                ? axisV + input.Shape.Rank
                : axisV;
            outShape[axis] = input.Shape[axis].IsFixed
                ? f(i, axis, input.Shape[axis].FixedValue)
                : Dimension.Unknown;
        }

        return outShape;
    }

    private IRType Visit(ITypeInferenceContext context, Slice target, TensorType input)
    {
        Shape outShape;
        if (input.Shape.IsRanked && input.Shape.Count == 0)
        {
            return new InvalidType("Slice Input should not scalar");
        }
        if (context.GetArgument(target, Slice.Axes) is TensorConst axes_con)
        {
            if (input.Shape.IsRanked)
            {
                if (context.GetArgument(target, Slice.Begins) is TensorConst begins_con &&
                    context.GetArgument(target, Slice.Ends) is TensorConst ends_con &&
                    context.GetArgument(target, Slice.Strides) is TensorConst strides_con)
                {
                    // end in onnx may be the maximum value of int64
                    // when use int, result value is -1
                    var ts_begins = begins_con.Value.Cast<long>();
                    var ts_ends = ends_con.Value.Cast<long>();
                    var ts_strides = strides_con.Value.Cast<long>();

                    outShape = ApplyAxis(axes_con, input, (i, axis, inDim) =>
                    {
                        var stride = ts_strides[i];

                        // reverse stride
                        if (stride < 0)
                        {
                            // document in onnx operators:
                            // for positive stepping and [0, dims[axes[i]]-1] for negative stepping.
                            var begin = System.Math.Clamp(ts_begins[i], 0L, inDim - 1);

                            // while for negative stepping it is clamped to [-1, dims[axes[i]]-1].
                            var end = System.Math.Clamp(ts_ends[i], -1L, inDim);
                            return (int)System.Math.Ceiling((float)System.Math.Abs(end - begin) /
                                                            System.Math.Abs(stride));
                        }
                        else
                        {
                            // starts[i] is clamped into the range [0, dims[axes[i]]]
                            var begin = ts_begins[i] < 0 ? inDim + ts_begins[i] : System.Math.Clamp(ts_begins[i], 0L, inDim);

                            // end[i] is clamped into the range [0, dims[axes[i]]]
                            var end = ts_ends[i] < 0 ? inDim + ts_ends[i] : System.Math.Clamp(ts_ends[i], 0L, inDim);
                            return (int)System.Math.Ceiling((float)System.Math.Abs(end - begin) /
                                                            System.Math.Abs(stride));
                        }
                    });
                    return input with { Shape = outShape };
                }
                else
                {
                    outShape = ApplyAxis(axes_con, input, (i, axis, inDim) => Dimension.Unknown);
                }
            }
            else
            {
                outShape = Shape.Unranked;
            }
        }
        else
        {
            return input with { Shape = Shape.Unknown(input.Shape.Rank) };
        }

        return input with { Shape = outShape };
    }
}
