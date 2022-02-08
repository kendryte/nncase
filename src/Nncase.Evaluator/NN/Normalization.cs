// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.NN;

/// <summary>
/// Evaluator for <see cref="BatchNormalization"/>.
/// </summary>
public class BatchNormalizationEvaluator : IEvaluator<BatchNormalization>, ITypeInferencer<BatchNormalization>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, BatchNormalization batchNorm)
    {
        var input = context.GetTorchArgumentValue(batchNorm, BatchNormalization.Input);
        var eps = context.GetArgumentValue(batchNorm, BatchNormalization.Epsilon);
        var mom = context.GetArgumentValue(batchNorm, BatchNormalization.Momentum);
        var m = torch.nn.BatchNorm2d(input.shape[^3], eps.ToScalar<float>(), mom.ToScalar<float>());
        return m.forward(input).ToConst();
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
    public Const Visit(IEvaluateContext context, InstanceNormalization i)
    {
        var input = context.GetTorchArgumentValue(i, InstanceNormalization.Input);
        var eps = context.GetArgumentValue(i, InstanceNormalization.Epsilon).ToScalar<float>();
        var f = torch.nn.InstanceNorm2d(input.shape[1], eps);
        return f.forward(input).ToConst();
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
    public Const Visit(IEvaluateContext context, LRN l)
    {
        var input = context.GetTorchArgumentValue(l, LRN.Input);
        var size = context.GetArgumentValueAsScalar<long>(l, LRN.Size);
        var alpha = context.GetArgumentValueAsScalar<float>(l, LRN.Alpha);
        var beta = context.GetArgumentValueAsScalar<float>(l, LRN.Beta);
        var k = context.GetArgumentValueAsScalar<float>(l, LRN.Bias);
        return torch.nn.LocalResponseNorm(size, alpha, beta, k).forward(input).ToConst();
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
