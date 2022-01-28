// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.NN;
using Tensorflow;
using static Tensorflow.Binding;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Hardmax"/>.
/// </summary>
public class HardmaxEvaluator : IEvaluator<Hardmax>, ITypeInferencer<Hardmax>
{
    /// <inheritdoc/>
    public Const Visit(EvaluatorContext context, Hardmax target)
    {
        // TODO: implement hardmax evaluator
        throw new NotImplementedException();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Hardmax target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Hardmax.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
