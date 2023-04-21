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
public class L2NormalizationEvaluator : IEvaluator<L2Normalization>, ITypeInferencer<L2Normalization>, ICostEvaluator<L2Normalization>
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

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="BatchNormalization"/>.
/// </summary>
public class BatchNormalizationEvaluator : IEvaluator<BatchNormalization>, ITypeInferencer<BatchNormalization>, ICostEvaluator<BatchNormalization>
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

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="InstanceNormalization"/>.
/// </summary>
public class InstanceNormalizationEvaluator : IEvaluator<InstanceNormalization>, ITypeInferencer<InstanceNormalization>, ICostEvaluator<InstanceNormalization>
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
        var input = context.CheckArgumentType<TensorType>(target, InstanceNormalization.Input);
        return Visit(input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, InstanceNormalization target)
    {
        var inputType = context.GetArgumentType<TensorType>(target, InstanceNormalization.Input);
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

/// <summary>
/// Evaluator for <see cref="LpNormalization"/>.
/// </summary>
public class LpNormalizationEvaluator : IEvaluator<LpNormalization>, ITypeInferencer<LpNormalization>, ICostEvaluator<LpNormalization>
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
