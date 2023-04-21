﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Slice"/>.
/// </summary>
public class SliceEvaluator : IEvaluator<Slice>, ITypeInferencer<Slice>, ICostEvaluator<Slice>
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
