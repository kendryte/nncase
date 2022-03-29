// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Slice"/>.
/// </summary>
public class SliceEvaluator : IEvaluator<Slice>, ITypeInferencer<Slice>
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
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Slice target, TensorType input)
    {
        if (context.GetArgument(target, Slice.Begins) is TensorConst begins_con &&
            context.GetArgument(target, Slice.Ends) is TensorConst ends_con &&
            context.GetArgument(target, Slice.Axes) is TensorConst axes_con &&
            context.GetArgument(target, Slice.Strides) is TensorConst strides_con)
        {
            var outShape = new List<Dimension>();
            var ts_begins = begins_con.Value.Cast<int>();
            var ts_ends = ends_con.Value.Cast<int>();
            var ts_strides = strides_con.Value.Cast<int>();

            // foreach (var axisV in axes_con.ToTensor<int>())
            var axesTensor = axes_con.Value.Cast<int>();
            for (int i = 0; i < axesTensor.Length; i++)
            {
                var axisV = axesTensor[i];
                var axis = axisV < 0
                    ? axisV + input.Shape.Rank
                    : axisV;
                var begin = ts_begins[i];
                var end = ts_ends[i];
                var stride = ts_strides[i];
                if (input.Shape[axis].IsFixed)
                {
                    var old = input.Shape[axis].FixedValue;
                    begin = begin >= 0 ? begin : old + begin;
                    end = end >= 0 ? end : old + begin;
                    stride = stride >= 0 ? stride : -stride;
                    outShape.Add((end - begin) / stride);
                }
                else
                {
                    outShape.Add(Dimension.Unknown);
                }
            }

            return input with { Shape = new Shape(outShape) };
        }

        return new InvalidType("Can't Infer Shape With Dynamic Input!");
    }
}
