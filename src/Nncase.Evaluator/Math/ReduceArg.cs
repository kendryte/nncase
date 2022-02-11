// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using TorchSharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Reduce"/>.
/// </summary>
public class ReduceArgEvaluator : IEvaluator<ReduceArg>, ITypeInferencer<ReduceArg>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, ReduceArg reduceArg)
    {
        var input = context.GetTorchArgumentValue(reduceArg, ReduceArg.Input);
        var axis = context.GetArgumentValue(reduceArg, ReduceArg.Axis).ToScalar<int>();
        var keepDims = context.GetArgumentValue(reduceArg, ReduceArg.KeepDims).ToScalar<bool>();
        var selectLastIndex = context.GetArgumentValue(reduceArg, ReduceArg.SelectLastIndex).ToScalar<bool>();
        if (selectLastIndex)
        {
            throw new NotImplementedException();
        }
        else
        {
            return (reduceArg.ReduceArgOp switch
            {
                ReduceArgOp.ArgMax => input.argmax(axis, keepDims),
                ReduceArgOp.ArgMin => input.argmin(axis, keepDims),
                _ => throw new ArgumentOutOfRangeException(nameof(reduceArg.ReduceArgOp)),
            }).ToConst();
        }
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ReduceArg target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ReduceArg.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, ReduceArg target, TensorType input)
    {
        if (context.GetArgument(target, ReduceArg.Axis) is Const axisValue &&
            context.GetArgument(target, ReduceArg.KeepDims) is Const keepDimsValue)
        {
            var shape = input.Shape.ToList();
            var axisIndex = axisValue.ToScalar<int>();
            axisIndex = axisIndex >= 0 ? axisIndex : input.Shape.Rank + axisIndex;
            if (keepDimsValue.ToScalar<bool>())
            {
                shape[axisIndex] = 1;
            }
            else
            {
                shape.RemoveAt(axisIndex);
            }

            return input with { Shape = new Shape(shape) };
        }
        else
        {
            return new InvalidType("ReduceArg axis and keepDims are not const");
        }
    }
}
