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
[EvaluatorGenerator]
[TypeInferGenerator]
public partial class CeluEvaluator : IEvaluator<Celu>, ITypeInferencer<Celu>, ICostEvaluator<Celu>, IMetricEvaluator<Celu>
{
    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Celu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public IValue Visit(IEvaluateContext context, Celu target)
    {
        var input = context.GetOrtArgumentValue(target, Celu.Input);
        var alpha = context.GetArgumentValueAsScalar<float>(target, Celu.Alpha);
        return Visit(input, alpha);
    }

    public Metric Visit(IMetricEvaluateContext context, Celu target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Celu.Input);
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) + CostUtility.GetMemoryAccess(inputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * (MetricUtility.ExpFLOPs + 6),
        };
    }

    private IValue Visit(OrtKISharp.Tensor input, float alpha)
    {
        return OrtKI.Celu(input, alpha).ToValue();
    }

    private TensorType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Elu"/>.
/// </summary>
[EvaluatorGenerator]
[TypeInferGenerator]
public partial class EluEvaluator : IEvaluator<Elu>, ITypeInferencer<Elu>, ICostEvaluator<Elu>, IMetricEvaluator<Elu>
{
    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Elu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Elu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * (MetricUtility.ExpFLOPs + 3),
        };
    }

    private IValue Visit(OrtKISharp.Tensor input, float alpha)
    {
        return OrtKI.Elu(input, alpha).ToValue();
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Elu"/>.
/// </summary>
public class HardSwishEvaluator : IEvaluator<HardSwish>, ITypeInferencer<HardSwish>, ICostEvaluator<HardSwish>, IMetricEvaluator<HardSwish>
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

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, HardSwish target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, HardSwish target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * 6,
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="LeakyRelu"/>.
/// </summary>
public class LeakyReluEvaluator : IEvaluator<LeakyRelu>, ITypeInferencer<LeakyRelu>, ICostEvaluator<LeakyRelu>, IShapeEvaluator<LeakyRelu>, IMetricEvaluator<LeakyRelu>
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

    public Cost Visit(ICostEvaluateContext context, LeakyRelu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, LeakyRelu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * 3,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, LeakyRelu target) => context.GetArgumentShape(target, LeakyRelu.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Relu"/>.
/// </summary>
public class ReluEvaluator : IEvaluator<Relu>, ITypeInferencer<Relu>, ICostEvaluator<Relu>, IShapeEvaluator<Relu>, IMetricEvaluator<Relu>
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

    public Cost Visit(ICostEvaluateContext context, Relu target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Relu.Input);
        return CostUtility.GetActivationCost(inputType, CostUtility.GetCPUCyclesOfMax());
    }

    public Metric Visit(IMetricEvaluateContext context, Relu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * 1,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Relu target) => context.GetArgumentShape(target, Relu.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Relu6"/>.
/// </summary>
public class Relu6Evaluator : IEvaluator<Relu6>, ITypeInferencer<Relu6>, ICostEvaluator<Relu6>, IMetricEvaluator<Relu6>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Relu6 relu6)
    {
        var input = context.GetOrtArgumentValue(relu6, Relu6.Input);
        return OrtKI.Clip(input, 0.0f, 6.0f).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Relu6 target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Relu6.Input);
        return Visit(input);
    }

    public Cost Visit(ICostEvaluateContext context, Relu6 target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, Relu6.Input);
        return CostUtility.GetActivationCost(inputType, CostUtility.GetCPUCyclesOfMax());
    }

    public Metric Visit(IMetricEvaluateContext context, Relu6 target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * 2,
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Selu"/>.
/// </summary>
public class SeluEvaluator : IEvaluator<Selu>, ITypeInferencer<Selu>, ICostEvaluator<Selu>, IMetricEvaluator<Selu>
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

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Selu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Selu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * (4 + MetricUtility.ExpFLOPs),
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Sigmoid"/>.
/// </summary>
public class SigmoidEvaluator : IEvaluator<Sigmoid>, ITypeInferencer<Sigmoid>, ICostEvaluator<Sigmoid>, IShapeEvaluator<Sigmoid>, IMetricEvaluator<Sigmoid>
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
    public Cost Visit(ICostEvaluateContext context, Sigmoid target)
    {
        var ret = context.GetReturnType<TensorType>();
        uint macPerElement = 3;
        return CostUtility.GetActivationCost(ret, macPerElement);
    }

    public Metric Visit(IMetricEvaluateContext context, Sigmoid target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * (3 + MetricUtility.ExpFLOPs),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Sigmoid target) => context.GetArgumentShape(target, Sigmoid.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="HardSigmoid"/>.
/// </summary>
public class HardSigmoidEvaluator : IEvaluator<HardSigmoid>, ITypeInferencer<HardSigmoid>, ICostEvaluator<HardSigmoid>, IMetricEvaluator<HardSigmoid>
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

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, HardSigmoid target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, HardSigmoid target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * 6,
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="PRelu"/>.
/// </summary>
public class PReluEvaluator : IEvaluator<PRelu>, ITypeInferencer<PRelu>, ICostEvaluator<PRelu>, IMetricEvaluator<PRelu>
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

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, PRelu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, PRelu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * 2,
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Erf"/>.
/// </summary>
public class ErfEvaluator : IEvaluator<Erf>, ITypeInferencer<Erf>, ICostEvaluator<Erf>, IShapeEvaluator<Erf>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Erf erf)
    {
        var input = context.GetOrtArgumentValue(erf, Erf.Input);
        return OrtKI.Erf(input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Erf target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Erf.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Erf target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Erf target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * 2,
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Erf target) => context.GetArgumentShape(target, Erf.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Sigmoid"/>.
/// </summary>
public class SwishEvaluator : IEvaluator<Swish>, ITypeInferencer<Swish>, ICostEvaluator<Swish>, IMetricEvaluator<Swish>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Swish swish)
    {
        var input = context.GetOrtArgumentValue(swish, Swish.Input);
        var beta = context.GetOrtArgumentValue(swish, Swish.Beta);
        return OrtKI.Mul(OrtKI.Sigmoid(input * beta), input).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Swish target)
    {
        var input = context.CheckArgumentType<IRType>(target, Swish.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Swish target)
    {
        var outputType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Swish target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * (MetricUtility.ExpFLOPs + 3),
        };
    }

    private IRType Visit(IRType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="Gelu"/>.
/// </summary>
public class GeluEvaluator : IEvaluator<Gelu>, ITypeInferencer<Gelu>, ICostEvaluator<Gelu>, IShapeEvaluator<Gelu>, IMetricEvaluator<Gelu>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Gelu gelu)
    {
        var input = context.GetOrtArgumentValue(gelu, Gelu.Input);
        var alpha = context.GetArgumentValueAsScalar<float>(gelu, Gelu.Alpha);

        var scaledInput = OrtKI.Mul(alpha, input);
        return OrtKI.Mul(0.5f, OrtKI.Mul(scaledInput, OrtKI.Add(OrtKI.Erf(OrtKI.Div(scaledInput, OrtKI.Sqrt(2f))), 1f))).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Gelu target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Gelu.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Gelu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, (CostUtility.GetCPUCyclesOfBinary(BinaryOp.Mul) * 3) + (CostUtility.GetCPUCyclesOfBinary(BinaryOp.Div) * 2) + CostUtility.GetCPUCyclesOfBinary(BinaryOp.Add)),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Gelu target)
    {
        var outputType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(outputType) * 2,
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType) * (MetricUtility.ExpFLOPs + MetricUtility.SqrtFLOPs + 6),
        };
    }

    public Expr Visit(IShapeEvaluateContext context, Gelu target) => context.GetArgumentShape(target, Gelu.Input);

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
