// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Clamp"/>.
/// </summary>
public class ClampEvaluator : IEvaluator<Clamp>, ITypeInferencer<Clamp>
{
    /// <inheritdoc/>
    public Const Visit(EvaluatorContext context, Clamp clamp)
    {
        var input = context.GetTorchArgument(clamp, Clamp.Input);
        var min = context.GetArgumentConst(clamp, Clamp.Min).ToArray<float>();
        var max = context.GetArgumentConst(clamp, Clamp.Max).ToArray<float>();
        return torch.clamp(input, min, max).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Clamp target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Clamp.Input);
        var min = context.CheckArgumentType<TensorType>(target, Clamp.Min);
        var max = context.CheckArgumentType<TensorType>(target, Clamp.Max);
        return Visit(input, min, max);
    }

    private IRType Visit(TensorType input, TensorType min, TensorType max)
    {
        return input;
    }
}
