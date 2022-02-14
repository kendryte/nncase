// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Celu"/>.
/// </summary>
public partial class CeluEvaluator : IEvaluator<Celu>, ITypeInferencer<Celu>
{
    /// <summary>
    /// visit the celu.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="alpha"></param>
    /// <returns></returns>
    public Const Visit(TorchSharp.torch.Tensor input, int alpha)
    {
        return input.celu().ToConst();
    }

    // /// <inheritdoc/>
    // public IRType Visit(ITypeInferenceContext context, Celu target)
    // {
    //     var input = context.CheckArgumentType<TensorType>(target, Celu.Input);
    //     return Visit(input);
    // }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Elu"/>.
/// </summary>
public class EluEvaluator : IEvaluator<Elu>, ITypeInferencer<Elu>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Elu elu)
    {
        var input = context.GetTorchArgumentValue(elu, Elu.Input);
        var alpha = context.GetArgumentValue(elu, Elu.Alpha).ToScalar<double>();
        return torchF.elu(input, alpha).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Elu target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Elu.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Elu"/>.
/// </summary>
public class HardSwishEvaluator : IEvaluator<HardSwish>, ITypeInferencer<HardSwish>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, HardSwish hardSwish)
    {
        var input = context.GetTorchArgumentValue(hardSwish, HardSwish.Input);
        return input.hardswish().ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, HardSwish target)
    {
        var input = context.CheckArgumentType<TensorType>(target, HardSwish.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="LeakyRelu"/>.
/// </summary>
public class LeakyReluEvaluator : IEvaluator<LeakyRelu>, ITypeInferencer<LeakyRelu>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, LeakyRelu leakyRelu)
    {
        var input = context.GetTorchArgumentValue(leakyRelu, LeakyRelu.Input);
        return input.leaky_relu(0.01).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, LeakyRelu target)
    {
        var input = context.CheckArgumentType<TensorType>(target, LeakyRelu.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Relu"/>.
/// </summary>
public class ReluEvaluator : IEvaluator<Relu>, ITypeInferencer<Relu>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Relu relu)
    {
        var input = context.GetTorchArgumentValue(relu, Relu.Input);
        return input.relu().ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Relu target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Relu.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Selu"/>.
/// </summary>
public class SeluEvaluator : IEvaluator<Selu>, ITypeInferencer<Selu>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Selu selu)
    {
        var input = context.GetTorchArgumentValue(selu, Selu.Input);
        return input.selu().ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Selu target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Selu.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Sigmoid"/>.
/// </summary>
public class SigmoidEvaluator : IEvaluator<Sigmoid>, ITypeInferencer<Sigmoid>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Sigmoid sigmoid)
    {
        var input = context.GetTorchArgumentValue(sigmoid, Sigmoid.Input);
        return input.sigmoid().ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Sigmoid target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Sigmoid.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
