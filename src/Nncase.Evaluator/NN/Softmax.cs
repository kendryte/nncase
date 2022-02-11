// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;
using Tensorflow;
using static Tensorflow.Binding;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="LogSoftmax"/>.
/// </summary>
public class LogSoftmaxEvaluator : IEvaluator<LogSoftmax>, ITypeInferencer<LogSoftmax>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, LogSoftmax logSoftMax)
    {
        var input = context.GetTorchArgumentValue(logSoftMax, LogSoftmax.Input);
        var dim = context.GetArgumentValue(logSoftMax, LogSoftmax.Axis).ToScalar<int>();
        return torchF.log_softmax(input, dim).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, LogSoftmax target)
    {
        var input = context.CheckArgumentType<TensorType>(target, LogSoftmax.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Softmax"/>.
/// </summary>
public class SoftmaxEvaluator : IEvaluator<Softmax>, ITypeInferencer<Softmax>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Softmax softMax)
    {
        var input = context.GetTorchArgumentValue(softMax, Softmax.Input);
        var dim = context.GetArgumentValue(softMax, Softmax.Axis).ToScalar<int>();
        return torchF.softmax(input, dim).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Softmax target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Softmax.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Softplus"/>.
/// </summary>
public class SoftplusEvaluator : IEvaluator<Softplus>, ITypeInferencer<Softplus>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Softplus softPlus)
    {
        var input = context.GetTorchArgumentValue(softPlus, Softplus.Input);
        return input.softplus().ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Softplus target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Softplus.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Softsign"/>.
/// </summary>
public class SoftsignEvaluator : IEvaluator<Softsign>, ITypeInferencer<Softsign>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Softsign softSign)
    {
        var input = context.GetTFArgumentValue(softSign, Softsign.Input);

        // Tensorflow.Net no this interface
        return tf.Context.ExecuteOp("Softsign", null, new ExecuteOpArgs(input))[0].ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Softsign target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Softsign.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
