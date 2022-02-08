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

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Expand"/>.
/// </summary>
public class ExpandEvaluator : IEvaluator<Expand>, ITypeInferencer<Expand>
{
    /// <inheritdoc/>
    public Const Visit(EvaluatorContext context, Expand expand)
    {
        var input = context.GetTorchArgument(expand, Expand.Input);
        if (input.shape.Length == 0)
        {
            input = input.reshape(1L);
        }

        var shape = context.GetArgumentConst(expand, Expand.Shape).ToArray<long>();

        // When the value of onnx is 1, the value of torch is -1
        var torchShape = shape.Select(x => x == 1 ? -1 : x).ToArray();
        if (torchShape.Length < input.shape.Length)
        {
            // [-1]*n.Concat(TorchShape)
            torchShape = Enumerable.Repeat(-1L, input.shape.Length - torchShape.Length).Concat(torchShape).ToArray();
        }

        return input.expand(torchShape).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Expand target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Cast.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Expand target, TensorType input)
    {
        if (context.GetArgument(target, Expand.Shape) is Const constShape)
        {
            return new TensorType(input.DType, constShape.ToArray<int>());
        }
        else
        {
            return new InvalidType("Expand Shape need const value");
        }
    }
}
