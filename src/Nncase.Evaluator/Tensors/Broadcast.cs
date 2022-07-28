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
        var shape = context.GetArgumentValueAsArray<int>(b, Broadcast.Shape);
        return input.BroadcastTo(shape).ToValue();
    }

    /// <inheritdoc/>
    public Cost? Visit(ICostEvaluateContext context, Broadcast target)
    {
        var input = context.GetArgumentType<TensorType>(target, Broadcast.Input);
        var ret = context.GetReturnType<TensorType>();
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(ret),
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    IRType Visit(TensorType Input, TensorType Shape, ITypeInferenceContext context, Broadcast op)
    {
        var shapeValue = context.GetArgument(op, Broadcast.Shape);
        if (shapeValue is TensorConst constShapeValue && Input.Shape.IsFixed)
        {
            return TypeInference.BroadcastType(Input, new TensorType(Input.DType, constShapeValue.Value.ToArray<int>()));
        }
        if (Shape.Shape[0].IsFixed)
        {
            return Input with { Shape = Enumerable.Repeat(Dimension.Unknown, Shape.Shape[0].FixedValue).ToArray() };
        }
        return Input with { Shape = IR.Shape.Unranked };
    }
}
