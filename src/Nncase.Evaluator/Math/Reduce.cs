// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Reduce"/>.
/// </summary>
public class ReduceEvaluator : IEvaluator<Reduce>, ITypeInferencer<Reduce>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Reduce reduce)
    {
        var input = context.GetOrtArgumentValue(reduce, Reduce.Input);
        var axis = context.GetArgumentValueAsArray<long>(reduce, Reduce.Axis);
        var keepDims = context.GetArgumentValueAsScalar<long>(reduce, Reduce.KeepDims);

        return (reduce.ReduceOp switch
        {
            ReduceOp.Mean => OrtKI.ReduceMean(input, axis, keepDims),
            ReduceOp.Max => OrtKI.ReduceMax(input, axis, keepDims),
            ReduceOp.Min => OrtKI.ReduceMin(input, axis, keepDims),
            ReduceOp.Prod => OrtKI.ReduceProd(input, axis, keepDims),
            ReduceOp.Sum => OrtKI.ReduceSum(
                input,
                context.GetInt64OrtTensorArgumentValue(reduce, Reduce.Axis),
                keepDims,
                0),
            _ => throw new ArgumentOutOfRangeException(),
        }).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Reduce target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Reduce.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Reduce target, TensorType input)
    {
        var args = context.GetArguments(target, Reduce.KeepDims, Reduce.Axis);
        return TypeInference.ReduceType(input, args[0], args[1]);
    }
}
