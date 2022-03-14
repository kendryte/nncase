// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Clamp"/>.
/// </summary>
public class ClampEvaluator : IEvaluator<Clamp>, ITypeInferencer<Clamp>, ICostEvaluator<Clamp>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Clamp clamp)
    {
        var input = context.GetTorchArgumentValue(clamp, Clamp.Input);
        var min = context.GetArgumentValueAsTensor<float>(clamp, Clamp.Min).ToTorchTensor();
        var max = context.GetArgumentValueAsTensor<float>(clamp, Clamp.Max).ToTorchTensor();
        return torch.clamp(input, min, max).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Clamp target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Clamp.Input);
        var min = context.CheckArgumentType<TensorType>(target, Clamp.Min);
        var max = context.CheckArgumentType<TensorType>(target, Clamp.Max);
        return Visit(input, min, max);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Clamp target)
    {
        var returnType = context.GetReturnType<TensorType>();
        var arithm = returnType.Shape.Prod().FixedValue;
        return new(arithm, arithm * returnType.DType.SizeInBytes);
    }

    private IRType Visit(TensorType input, TensorType min, TensorType max)
    {
        return input;
    }
}
