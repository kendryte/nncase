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
        Cost cost;
        switch (inType, returnType)
        {
            case (TensorType tensorType, DistributedType distTensorType):
                var partedOutType = DistributedUtilities.GetDividedTensorType(distTensorType, out _);
                cost = new Cost()
                {
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(tensorType),
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(partedOutType),
                };
                break;
            case (DistributedType distTensorType, TensorType tensorType):
                var partedInType = DistributedUtilities.GetDividedTensorType(distTensorType, out _);
                cost = new Cost()
                {
                    [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(partedInType),
                    [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(tensorType),
                };
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(context));
        }

        return cost;
    }
}
