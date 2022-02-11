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
    public IValue Visit(IEvaluateContext context, LogSoftmax logSoftMax)
    {
        var input = context.GetTorchArgumentValue(logSoftMax, LogSoftmax.Input);
        var dim = context.GetArgumentValueAsScalar<int>(logSoftMax, LogSoftmax.Axis);
        return torchF.log_softmax(input, dim).ToValue();
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
    public IValue Visit(IEvaluateContext context, Softmax softMax)
    {
        var input = context.GetTorchArgumentValue(softMax, Softmax.Input);
        var dim = context.GetArgumentValueAsScalar<int>(softMax, Softmax.Axis);
        return torchF.softmax(input, dim).ToValue();
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
    public IValue Visit(IEvaluateContext context, Softplus softPlus)
    {
        var input = context.GetTorchArgumentValue(softPlus, Softplus.Input);
        return input.softplus().ToValue();
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
    public IValue Visit(IEvaluateContext context, Softsign softSign)
    {
        var input = context.GetTFArgumentValue(softSign, Softsign.Input);

        // Tensorflow.Net no this interface
        return tf.Context.ExecuteOp("Softsign", null!, new ExecuteOpArgs(input))[0].ToValue();
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
