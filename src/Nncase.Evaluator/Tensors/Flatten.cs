// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Flatten"/>.
/// </summary>
public class FlattenEvaluator : IEvaluator<Flatten>, ITypeInferencer<Flatten>, ICostEvaluator<Flatten>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Flatten flatten)
    {
        var input = context.GetOrtArgumentValue(flatten, Flatten.Input);
        var dim = context.GetArgumentValueAsScalar<int>(flatten, Flatten.Axis);
        return OrtKI.Flatten(input, dim).ToValue();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Flatten target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Flatten.Input);
        return Visit(context, target, input);
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Flatten target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    private IRType Visit(ITypeInferenceContext context, Flatten target, TensorType input)
    {
        if (context.GetArgument(target, Flatten.Axis) is TensorConst axisV)
        {
            if (input.Shape.IsFixed)
            {
                var axisValue = Util.PositiveIndex(axisV.Value.ToScalar<int>(), input);
                var first = input.Shape.Take(axisValue).Aggregate(1, (x, y) => x * y.FixedValue);
                var second = input.Shape.Take(axisValue..input.Shape.Count).Aggregate(1, (x, y) => x * y.FixedValue);
                return input with { Shape = new[] { first, second } };
            }
        }

        return input with { Shape = Shape.Unknown(2) };
    }
}
