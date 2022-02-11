// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Reduce"/>.
/// </summary>
public class ReduceEvaluator : IEvaluator<Reduce>, ITypeInferencer<Reduce>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Reduce reduce)
    {
        var input = context.GetTFArgumentValue(reduce, Reduce.Input);
        var axis = context.GetArgumentValueAsArray<long>(reduce, Reduce.Axis);
        var keepDims = context.GetArgumentValueAsScalar<bool>(reduce, Reduce.KeepDims);

        return (reduce.ReduceOp switch
        {
            ReduceOp.Mean => tf.reduce_mean(input, axis, keepDims),
            ReduceOp.Max => tf.reduce_max(input, axis, keepDims),
            ReduceOp.Min => tf.reduce_min(input, axis, keepDims),
            ReduceOp.Prod => tf.reduce_prod(input, axis, keepDims),
            ReduceOp.Sum => tf.reduce_sum(input, axis, keepdims: keepDims),
            _ => throw new ArgumentOutOfRangeException(),
        }).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Reduce target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Reduce.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Reduce target, TensorType input)
    {
        var args = context.GetArguments(target, Reduce.Axis, Reduce.KeepDims);
        return TypeInference.ReduceType(input, args[0], args[1]);
    }
}
