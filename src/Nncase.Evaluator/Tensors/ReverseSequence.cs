// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;
using Tensorflow;
using static Tensorflow.Binding;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="ReverseSequence"/>.
/// </summary>
public class ReverseSequenceEvaluator : IEvaluator<ReverseSequence>, ITypeInferencer<ReverseSequence>
{
    /// <inheritdoc/>
    public Const Visit(EvaluatorContext context, ReverseSequence random)
    {
        var input = context.GetTFArgument(random, ReverseSequence.Input);
        var seqLens = context.GetTFArgument(random, ReverseSequence.SeqLens);
        var batchAxis = context.GetArgumentConstScalar<int>(random, ReverseSequence.BatchAxis);
        var timeAxis = context.GetArgumentConstScalar<int>(random, ReverseSequence.TimeAxis);
        return tf.Context.ExecuteOp(
            "ReverseSequence",
            null!,
            new ExecuteOpArgs(input, seqLens)
                .SetAttributes(new
                {
                    seq_dim = timeAxis,
                    batch_dim = batchAxis,
                }))[0].ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, ReverseSequence target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Reshape.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, ReverseSequence target, TensorType input)
    {
        return input;
    }
}
