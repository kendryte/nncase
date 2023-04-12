// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Broadcast"/>.
/// </summary>
[TypeInferGenerator]
public sealed partial class BroadcastEvaluator : IEvaluator<Broadcast>, ITypeInferencer<Broadcast>, ICostEvaluator<Broadcast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Broadcast b)
    {
        var input = context.GetOrtArgumentValue(b, Broadcast.Input);
        var shape = context.GetArgumentValueAsArray<long>(b, Broadcast.Shape);
        return input.BroadcastTo(shape).ToValue();
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Broadcast target)
    {
        var input = context.GetArgumentType<TensorType>(target, Broadcast.Input);
        var ret = context.GetReturnType<TensorType>();
        return CostUtility.GetBroadcastCost(input, ret);
    }

    private IRType Visit(TensorType input, TensorType shape, ITypeInferenceContext context, Broadcast op)
    {
        var shapeValue = context.GetArgument(op, Broadcast.Shape);
        if (shapeValue is TensorConst constShapeValue && input.Shape.IsFixed)
        {
            return TypeInference.BroadcastType(input, new TensorType(input.DType, constShapeValue.Value.ToArray<int>()));
        }

        if (shape.Shape[0].IsFixed)
        {
            return input with { Shape = Enumerable.Repeat(Dimension.Unknown, shape.Shape[0].FixedValue).ToArray() };
        }

        return input with { Shape = IR.Shape.Unranked };
    }
}
