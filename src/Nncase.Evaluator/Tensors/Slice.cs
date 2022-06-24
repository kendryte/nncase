// Copyright (c) Canaan Inc. All rights reserved.
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
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost? Visit(ICostEvaluateContext context, Slice target)
    {
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    private IRType Visit(ITypeInferenceContext context, Slice target, TensorType input)
    {
        if (context.GetArgument(target, Slice.Begins) is TensorConst begins_con &&
            context.GetArgument(target, Slice.Ends) is TensorConst ends_con &&
            context.GetArgument(target, Slice.Axes) is TensorConst axes_con &&
            context.GetArgument(target, Slice.Strides) is TensorConst strides_con)
        {
            // end in onnx may be the maximum value of int64
            // when use int, result value is -1
            var outShape = input.Shape.ToArray();
            var ts_begins = begins_con.Value.Cast<long>();
            var ts_ends = ends_con.Value.Cast<long>();
            var ts_strides = strides_con.Value.Cast<long>();

            var axesTensor = axes_con.Value.Cast<int>();
            for (int i = 0; i < axesTensor.Length; i++)
            {
                var axisV = axesTensor[i];
                var axis = axisV < 0
                    ? axisV + input.Shape.Rank
                    : axisV;
                var begin = ts_begins[i];
                var end = System.Math.Min(ts_ends[i], input.Shape[axis].FixedValue);
                var stride = ts_strides[i];
                if (input.Shape[axis].IsFixed)
                {
                    outShape[axis] =
                        (int) System.Math.Ceiling((float) System.Math.Abs(end - begin) / System.Math.Abs(stride));
                }
                else
                {
                    outShape[axis] = Dimension.Unknown;
                }
            }

            return input with { Shape = new Shape(outShape) };
        }

        return input with { Shape = Shape.Unranked };
    }
}
