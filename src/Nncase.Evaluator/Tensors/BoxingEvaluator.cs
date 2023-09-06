// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Tensors;

public sealed class BoxingEvaluator : ITypeInferencer<Boxing>, ICostEvaluator<Boxing>
{
    public IRType Visit(ITypeInferenceContext context, Boxing target)
    {
        return target.NewType;
    }

    public Cost Visit(ICostEvaluateContext context, Boxing target)
    {
        var inType = context.GetArgumentType<IRType>(target, Boxing.Input);
        var returnType = context.GetReturnType<IRType>();

        return (inType, returnType) switch
        {
            (TensorType tensorType, DistTensorType distTensorType) => Visit(tensorType, distTensorType),
            (DistTensorType distTensorType, TensorType tensorType) => Visit(tensorType, distTensorType),
            _ => throw new NotImplementedException(),
        };
    }

    /// <summary>
    /// calc the cost for load/store to distribute.
    /// </summary>
    /// <param name="tensorType">device tensor.</param>
    /// <param name="distTensorType">distribute tensor.</param>
    /// <returns>cost.</returns>
    private Cost Visit(TensorType tensorType, DistTensorType distTensorType)
    {
        var shape = distTensorType.TensorType.Shape.ToValueArray();
        foreach (var axis in distTensorType.NdSbp.OfType<SBPSplit>().Select(s => s.Axis))
        {
            shape[axis] /= axis;
        }

        return new Cost()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(tensorType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(distTensorType.TensorType with { Shape = shape }),
        };
    }
}
