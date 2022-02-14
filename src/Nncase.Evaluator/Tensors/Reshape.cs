// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using TorchSharp;
using Range = Nncase.IR.Tensors.Range;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Range"/>.
/// </summary>
public class ReshapeEvaluator : IEvaluator<Reshape>, ITypeInferencer<Reshape>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Reshape reshape)
    {
        var input = context.GetTorchArgumentValue(reshape, Reshape.Input);
        var shape = context.GetArgumentValueAsTensor<long>(reshape, Reshape.Shape).ToArray<long>();
        return input.reshape(shape).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Reshape target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Reshape.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Reshape target, TensorType input)
    {
        if (context.GetArgument(target, Reshape.Shape) is TensorConst shape_con)
        {
            return input with { Shape = new Shape(shape_con.Value.Cast<int>()) };
        }

        return input with { Shape = Shape.Unranked };
    }
}
