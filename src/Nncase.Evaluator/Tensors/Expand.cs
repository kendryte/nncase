// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using OrtKISharp;
using TorchSharp;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Expand"/>.
/// </summary>
public class ExpandEvaluator : IEvaluator<Expand>, ITypeInferencer<Expand>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Expand expand)
    {
        var input = context.GetOrtArgumentValue(expand, Expand.Input);
        var shape = context.GetOrtArgumentValue(expand, Expand.Shape);
        return OrtKI.Expand(input, shape).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Expand target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Expand.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Expand target, TensorType input)
    {
        if (context.GetArgument(target, Expand.Shape) is TensorConst constShape)
        {
            return new TensorType(input.DType, new Shape(constShape.Value.Cast<int>()));
        }
        else
        {
            return new InvalidType("Expand Shape need const value");
        }
    }
}
