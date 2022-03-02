// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="BatchNormalization"/>.
/// </summary>
public class BatchNormalizationEvaluator : IEvaluator<BatchNormalization>, ITypeInferencer<BatchNormalization>
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
        return Visit(input);
    }

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="InstanceNormalization"/>.
/// </summary>
public class InstanceNormalizationEvaluator : IEvaluator<InstanceNormalization>, ITypeInferencer<InstanceNormalization>
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

    private IRType Visit(TensorType input)
    {
        return input;
    }
}

/// <summary>
/// Evaluator for <see cref="LRN"/>.
/// </summary>
public class LRNEvaluator : IEvaluator<LRN>, ITypeInferencer<LRN>
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

    private IRType Visit(TensorType input)
    {
        return input;
    }
}
