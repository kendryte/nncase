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
/// Evaluator for <see cref="Flatten"/>.
/// </summary>
public class FlattenEvaluator : IEvaluator<Flatten>, ITypeInferencer<Flatten>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Flatten flatten)
    {
        var input = context.GetTorchArgumentValue(flatten, Flatten.Input);
        var dim = context.GetArgumentValue(flatten, Flatten.Axis).ToScalar<int>();
        var v = torch.nn.Flatten(0, dim);
        return v.forward(input).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Flatten target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Cast.Input);
        return Visit(context, target, input);
    }

    private IRType Visit(ITypeInferenceContext context, Flatten target, TensorType input)
    {
        if (context.GetArgument(target, Flatten.Axis) is Const axisV)
        {
            if (input.Shape.IsFixed)
            {
                var axisValue = axisV.ToScalar<int>();
                var first = input.Shape.Take(axisValue).Aggregate(1, (x, y) => x * y.FixedValue);
                var second = input.Shape.Take(axisValue..input.Shape.Count).Aggregate(1, (x, y) => x * y.FixedValue);
                return input with { Shape = new[] { first, second } };
            }
            else
            {
                return new InvalidType("Can't infer shape with dynamic input in Flatten");
            }
        }
        else
        {
            return new InvalidType("Can't infer shape with dynamic axis in Flatten");
        }
    }
}
