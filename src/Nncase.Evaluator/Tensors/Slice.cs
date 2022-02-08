// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Slice"/>.
/// </summary>
public class SliceEvaluator : IEvaluator<Slice>, ITypeInferencer<Slice>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Slice sl)
    {
        var input = context.GetTorchArgumentValue(sl, Slice.Input);
        var begins = context.GetTorchArgumentValue(sl, Slice.Begins);
        var ends = context.GetTorchArgumentValue(sl, Slice.Ends);
        var axes = context.GetArgumentValueAsArray<int>(sl, Slice.Axes)
            .Select(x => x < 0 ? x + input.shape.Rank : x);
        var strides = context.GetTorchArgumentValue(sl, Slice.Strides);
        var axesIndex = 0;
        var indices = Enumerable.Range(0, input.shape.Length).Select(i =>
            axes.Contains(i) ?
                torch.TensorIndex.Slice(
                    GetItem(begins, axesIndex),
                    GetItem(ends, axesIndex),
                    GetItem(strides, axesIndex++)) :
                torch.TensorIndex.Slice()
        ).ToArray();
        return input[indices].ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Slice target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Slice.Input);
        return Visit(context, target, input);
    }

    private static long GetItem(torch.Tensor tensor, int index)
    {
        if (tensor.shape.Rank != 1)
        {
            throw new NotSupportedException("Unsupported Rank which > 1 in GetItem(tensor, index)");
        }

        return tensor.to_type(torch.ScalarType.Int64).ReadCpuInt64(index);
    }

    private IRType Visit(ITypeInferenceContext context, Slice target, TensorType input)
    {
        if (context.GetArgument(target, Slice.Begins) is Const begins_con &&
            context.GetArgument(target, Slice.Ends) is Const ends_con &&
            context.GetArgument(target, Slice.Axes) is Const axes_con &&
            context.GetArgument(target, Slice.Strides) is Const strides_con)
        {
            var outShape = new List<Dimension>();
            var ts_begins = begins_con.ToTensor<int>();
            var ts_ends = ends_con.ToTensor<int>();
            var ts_strides = strides_con.ToTensor<int>();

            // foreach (var axisV in axes_con.ToTensor<int>())
            var axesTensor = axes_con.ToTensor<int>();
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
