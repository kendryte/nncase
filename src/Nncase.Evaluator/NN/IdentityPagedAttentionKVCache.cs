// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

public sealed class IdentityPagedAttentionKVCacheEvaluator : ITypeInferencer<IdentityPagedAttentionKVCache>, ICostEvaluator<IdentityPagedAttentionKVCache>, IEvaluator<IdentityPagedAttentionKVCache>
{
    public IRType Visit(ITypeInferenceContext context, IdentityPagedAttentionKVCache target)
    {
        var input = context.CheckArgumentType<IRType>(target, IdentityPagedAttentionKVCache.Input);
        switch (input)
        {
            case DistributedType distType:
                if (distType.Placement.Name == "cdyxt")
                {
                    // for kv cache.
                    if (distType.TensorType.Shape.Rank > 3 &&
                        distType.AxisPolicies[0] is SBPSplit { Axes: [1] } &&
                        distType.AxisPolicies[1] is SBPSplit { Axes: [2, 3] } &&
                        distType.AxisPolicies.Skip(2).All(x => x is SBPBroadCast))
                    {
                        return distType;
                    }
                    else if (distType.TensorType.Shape.Rank == 3)
                    {
                        return distType;
                    }
                }

                return new InvalidType("IdentityPagedAttentionKVCache only support input with cdyxt placement and split policy");
            case TensorType inputTType:
                return inputTType;
            default:
                return input;
        }
    }

    public Cost Visit(ICostEvaluateContext context, IdentityPagedAttentionKVCache target)
    {
        var input = context.GetArgumentType<IRType>(target, IdentityPagedAttentionKVCache.Input);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(input),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(input),
        };
    }

    public IValue Visit(IEvaluateContext context, IdentityPagedAttentionKVCache target)
    {
        var input = context.GetArgumentValue(target, IdentityPagedAttentionKVCache.Input);
        return input;
    }
}
