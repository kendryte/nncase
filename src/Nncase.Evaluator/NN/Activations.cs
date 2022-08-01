// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="Celu"/>.
/// </summary>
[EvaluatorGenerator, TypeInferGenerator]
public partial class CeluEvaluator : IEvaluator<Celu>, ITypeInferencer<Celu>
{
    private IValue Visit(OrtKISharp.Tensor Input, float Alpha)
    {
        return OrtKI.Celu(Input, Alpha).ToValue();
    }

    private TensorType Visit(TensorType Input)
    {
        return Input;
    }
}

/// <summary>
/// Evaluator for <see cref="Elu"/>.
/// </summary>
[EvaluatorGenerator, TypeInferGenerator]
public partial class EluEvaluator : IEvaluator<Elu>, ITypeInferencer<Elu>
{
    IValue Visit(OrtKISharp.Tensor Input, float Alpha)
    {
        return OrtKI.Elu(Input, Alpha).ToValue();
    }

    IRType Visit(TensorType Input)
    {
        return Input;
    }
}

/// <summary>
/// Evaluator for <see cref="Elu"/>.
/// </summary>
public class HardSwishEvaluator : IEvaluator<HardSwish>, ITypeInferencer<HardSwish>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, HardSwish hardSwish)
    {
        // onnxruntime no hardswish impl
        var input = context.GetOrtArgumentValue(hardSwish, HardSwish.Input);
        return (input * OrtKI.HardSigmoid(input, 1 / 6f, 0.5f)).ToValue();
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
    public IValue Visit(IEvaluateContext context, LeakyRelu leakyRelu)
    {
        var input = context.GetOrtArgumentValue(leakyRelu, LeakyRelu.Input);
        var alpha = context.GetArgumentValueAsScalar<float>(leakyRelu, LeakyRelu.Alpha);
        return OrtKI.LeakyRelu(input, alpha).ToValue();
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
    public IValue Visit(IEvaluateContext context, Relu relu)
    {
        var input = context.GetOrtArgumentValue(relu, Relu.Input);
        return OrtKI.Relu(input).ToValue();
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
    public IValue Visit(IEvaluateContext context, Selu selu)
    {
        var input = context.GetOrtArgumentValue(selu, Selu.Input);
        var alpha = context.GetArgumentValueAsScalar<float>(selu, Selu.Alpha);
        var gamma = context.GetArgumentValueAsScalar<float>(selu, Selu.Gamma);
        return OrtKI.Selu(input, alpha, gamma).ToValue();
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
public class SigmoidEvaluator : IEvaluator<Sigmoid>, ITypeInferencer<Sigmoid>, ICostEvaluator<Sigmoid>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Sigmoid sigmoid)
    {
        var input = context.GetOrtArgumentValue(sigmoid, Sigmoid.Input);
        return OrtKI.Sigmoid(input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Sigmoid target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Sigmoid.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost? Visit(ICostEvaluateContext context, Sigmoid target)
    {
        var ret = context.GetReturnType<TensorType>();
        var macPerElement = 3;
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(ret, macPerElement),
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="HardSigmoid"/>.
/// </summary>
public class HardSigmoidEvaluator : IEvaluator<HardSigmoid>, ITypeInferencer<HardSigmoid>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, HardSigmoid sigmoid)
    {
        var input = context.GetOrtArgumentValue(sigmoid, HardSigmoid.Input);
        var alpha = context.GetArgumentValueAsScalar<float>(sigmoid, HardSigmoid.Alpha);
        var beta = context.GetArgumentValueAsScalar<float>(sigmoid, HardSigmoid.Beta);
        return OrtKI.HardSigmoid(input, alpha, beta).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, HardSigmoid target)
    {
        var input = context.CheckArgumentType<TensorType>(target, HardSigmoid.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="PRelu"/>.
/// </summary>
public class PReluEvaluator : IEvaluator<PRelu>, ITypeInferencer<PRelu>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, PRelu sigmoid)
    {
        var input = context.GetOrtArgumentValue(sigmoid, PRelu.Input);
        var slope = context.GetOrtArgumentValue(sigmoid, PRelu.Slope);
        return OrtKI.PRelu(input, slope).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, PRelu target)
    {
        var input = context.CheckArgumentType<TensorType>(target, PRelu.Input);
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
