// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Broadcast"/>.
/// </summary>
public class BroadcastEvaluator : IEvaluator<Broadcast>, ITypeInferencer<Broadcast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Broadcast b)
    {
        var input = context.GetOrtArgumentValue(b, Broadcast.Input);
        var shape = context.GetArgumentValueAsArray<int>(b, Broadcast.Shape);
        return input.BroadcastTo(shape).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Broadcast target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Broadcast.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Broadcast target, TensorType input)
    {
        var shapeValue = context.GetArgument(target, Broadcast.Shape);
        if (shapeValue is TensorConst constShapeValue && input.Shape.IsFixed)
        {
            return TypeInference.BroadcastType(input, new TensorType(input.DType, constShapeValue.Value.ToArray<int>()));
        }
        else
        {
            return new InvalidType("Broadcast shape is unknown");
        }
    }
}
