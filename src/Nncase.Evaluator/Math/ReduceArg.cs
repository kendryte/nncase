// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Reduce"/>.
/// </summary>
public class ReduceArgEvaluator : IEvaluator<ReduceArg>, ITypeInferencer<ReduceArg>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, ReduceArg reduceArg)
    {
        var input = context.GetOrtArgumentValue(reduceArg, ReduceArg.Input);
        var axis = context.GetArgumentValueAsScalar<long>(reduceArg, ReduceArg.Axis);
        var keepDims = context.GetArgumentValueAsScalar<long>(reduceArg, ReduceArg.KeepDims);
        var selectLastIndex = context.GetArgumentValueAsScalar<long>(reduceArg, ReduceArg.SelectLastIndex);
        return (reduceArg.ReduceArgOp switch
        {
            ReduceArgOp.ArgMax => OrtKI.ArgMax(input, axis, keepDims, selectLastIndex),
            ReduceArgOp.ArgMin => OrtKI.ArgMin(input, axis, keepDims, selectLastIndex),
            _ => throw new ArgumentOutOfRangeException(nameof(reduceArg.ReduceArgOp)),
        }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ReduceArg target)
    {
        var input = context.CheckArgumentType<TensorType>(target, ReduceArg.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, ReduceArg target, TensorType input)
    {
        if (context.GetArgument(target, ReduceArg.Axis) is TensorConst axisValue &&
            context.GetArgument(target, ReduceArg.KeepDims) is TensorConst keepDimsValue)
        {
            var shape = input.Shape.ToList();
            var axisIndex = axisValue.Value.ToScalar<int>();
            axisIndex = axisIndex >= 0 ? axisIndex : input.Shape.Rank + axisIndex;
            if (keepDimsValue.Value.ToScalar<bool>())
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
