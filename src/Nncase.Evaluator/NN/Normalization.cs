// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="L2Normalization"/>.
/// </summary>
public class L2NormalizationEvaluator : IEvaluator<L2Normalization>, ITypeInferencer<L2Normalization>, ICostEvaluator<L2Normalization>, IMetricEvaluator<L2Normalization>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, L2Normalization norm)
    {
        var input = context.GetOrtArgumentValue(norm, L2Normalization.Input);
        var square = input * input;

        var size = input.Rank == 1 ? 1 : input.Rank - 1;
        var axes = new long[size];
        if (size == 1)
        {
            axes[0] = 0;
        }
        else
        {
            for (int i = 1; i <= size; i++)
            {
                axes[i - 1] = i;
            }
        }

        var sum = OrtKI.ReduceSum(square, axes, 1, 0);
        var max = OrtKI.Max(new[] { sum, 1e-10F });
        var sqrt = OrtKI.Sqrt(max);
        var div = input / sqrt;
        return div.ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, L2Normalization target)
    {
        var input = context.CheckArgumentType<TensorType>(target, L2Normalization.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, L2Normalization target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, L2Normalization.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, L2Normalization target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, L2Normalization.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="BatchNormalization"/>.
/// </summary>
public class BatchNormalizationEvaluator : IEvaluator<BatchNormalization>, ITypeInferencer<BatchNormalization>, ICostEvaluator<BatchNormalization>, IMetricEvaluator<BatchNormalization>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, BatchNormalization batchNorm)
    {
        var input = context.GetOrtArgumentValue(batchNorm, BatchNormalization.Input);
        var scale = context.GetOrtArgumentValue(batchNorm, BatchNormalization.Scale);
        var bias = context.GetOrtArgumentValue(batchNorm, BatchNormalization.Bias);
        var inputMean = context.GetOrtArgumentValue(batchNorm, BatchNormalization.InputMean);
        var inputVar = context.GetOrtArgumentValue(batchNorm, BatchNormalization.InputVar);
        var eps = context.GetArgumentValueAsScalar<float>(batchNorm, BatchNormalization.Epsilon);
        var mom = context.GetArgumentValueAsScalar<float>(batchNorm, BatchNormalization.Momentum);
        return OrtKI.BatchNormalization(input, scale, bias, inputMean, inputVar, eps, mom).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, BatchNormalization target)
    {
        var input = context.CheckArgumentType<TensorType>(target, BatchNormalization.Input);
        context.CheckArgumentType<TensorType>(target, BatchNormalization.Scale);
        context.CheckArgumentType<TensorType>(target, BatchNormalization.Bias);
        context.CheckArgumentType<TensorType>(target, BatchNormalization.InputVar);
        context.CheckArgumentType<TensorType>(target, BatchNormalization.InputMean);
        context.CheckArgumentType<TensorType>(target, BatchNormalization.Epsilon);

        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, BatchNormalization target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, BatchNormalization.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, BatchNormalization target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, BatchNormalization.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(returnType, 5 + (int)MetricUtility.SqrtFLOPs),
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="InstanceNormalization"/>.
/// </summary>
public class InstanceNormalizationEvaluator : IEvaluator<InstanceNormalization>, ITypeInferencer<InstanceNormalization>, ICostEvaluator<InstanceNormalization>, IMetricEvaluator<InstanceNormalization>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, InstanceNormalization i)
    {
        var input = context.GetOrtArgumentValue(i, InstanceNormalization.Input);
        var scale = context.GetOrtArgumentValue(i, InstanceNormalization.Scale);
        var bias = context.GetOrtArgumentValue(i, InstanceNormalization.Bias);
        var eps = context.GetArgumentValueAsScalar<float>(i, InstanceNormalization.Epsilon);
        return OrtKI.InstanceNormalization(input, scale, bias, eps).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, InstanceNormalization target)
    {
        var input = context.CheckArgumentType<IRType>(target, InstanceNormalization.Input);
        var scale = context.CheckArgumentType<IRType>(target, InstanceNormalization.Scale);
        var bias = context.CheckArgumentType<IRType>(target, InstanceNormalization.Bias);
        return (input, scale, bias) switch
        {
            (DistributedType a, DistributedType b, DistributedType c) => Visit(a, b, c),
            (TensorType a, TensorType, TensorType) => Visit(a),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, InstanceNormalization target)
    {
        var inputType = context.GetArgumentType<IRType>(target, InstanceNormalization.Input);
        var returnType = context.GetReturnType<IRType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, InstanceNormalization target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, InstanceNormalization.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }

    private IRType Visit(DistributedType input, DistributedType scale, DistributedType bias)
    {
        var invalid = new InvalidType($"{input}, {scale}, {bias} not support");
        if (input.Placement != scale.Placement || scale.Placement != bias.Placement)
        {
            return invalid;
        }

        var ndsbp = new SBP[input.Placement.Rank];

        // scale & bias always on Channel
        const int rAxis = 1;

        for (int i = 0; i < input.Placement.Rank; i++)
        {
            switch (input.NdSBP[i], scale.NdSBP[i], bias.NdSBP[i])
            {
                case (SBPSplit { Axis: int ix }, SBPSplit { Axis: int sx }, SBPSplit { Axis: int bx }) when ix == rAxis && sx == (ix - rAxis) && bx == sx:
                    ndsbp[i] = SBP.S(ix);
                    break;
                case (SBPSplit { Axis: int ix }, SBPBroadCast, SBPBroadCast) when ix != rAxis:
                    ndsbp[i] = SBP.S(ix);
                    break;
                case (SBPBroadCast, SBPBroadCast, SBPBroadCast):
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return invalid;
            }
        }

        return new DistributedType(input.TensorType, ndsbp, input.Placement);
    }
}

/// <summary>
/// Evaluator for <see cref="LpNormalization"/>.
/// </summary>
public class LpNormalizationEvaluator : IEvaluator<LpNormalization>, ITypeInferencer<LpNormalization>, ICostEvaluator<LpNormalization>, IMetricEvaluator<LpNormalization>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, LpNormalization norm)
    {
        var input = context.GetOrtArgumentValue(norm, LpNormalization.Input);
        var axis = context.GetArgumentValueAsScalar<long>(norm, LpNormalization.Axis);
        var p = context.GetArgumentValueAsScalar<long>(norm, LpNormalization.P);

        return OrtKI.LpNormalization(input, axis, p).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, LpNormalization target)
    {
        var input = context.CheckArgumentType<TensorType>(target, LpNormalization.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, LpNormalization target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, LpNormalization.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, LpNormalization target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, LpNormalization.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(inputType) + CostUtility.GetMemoryAccess(returnType),
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="LRN"/>.
/// </summary>
public class LRNEvaluator : IEvaluator<LRN>, ITypeInferencer<LRN>, ICostEvaluator<LRN>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, LRN l)
    {
        var input = context.GetOrtArgumentValue(l, LRN.Input);
        var size = context.GetArgumentValueAsScalar<long>(l, LRN.Size);
        var alpha = context.GetArgumentValueAsScalar<float>(l, LRN.Alpha);
        var beta = context.GetArgumentValueAsScalar<float>(l, LRN.Beta);
        var bias = context.GetArgumentValueAsScalar<float>(l, LRN.Bias);
        return OrtKI.LRN(input, alpha, beta, bias, size).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, LRN target)
    {
        var input = context.CheckArgumentType<TensorType>(target, LRN.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, LRN target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, LRN.Input);
        var returnType = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(inputType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(returnType),
        };
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
