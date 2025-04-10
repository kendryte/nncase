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

public sealed class UpdatePagedAttentionKVCacheEvaluator : ITypeInferencer<UpdatePagedAttentionKVCache>, ICostEvaluator<UpdatePagedAttentionKVCache>, IEvaluator<UpdatePagedAttentionKVCache>
{
    public IRType Visit(ITypeInferenceContext context, UpdatePagedAttentionKVCache target)
    {
        var slots = context.CheckArgumentType<IRType>(target, UpdatePagedAttentionKVCache.Slots);
        var kvCache = context.CheckArgumentType<IRType>(target, UpdatePagedAttentionKVCache.KVCache);
        return slots switch
        {
            DistributedType dslots => Visit(context, target, dslots, kvCache),
            TensorType tslots => Visit(context, target, tslots, kvCache),
            _ => new InvalidType("not support type"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, UpdatePagedAttentionKVCache target)
    {
        var slotsType = context.GetArgumentType<IRType>(target, UpdatePagedAttentionKVCache.Slots);
        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(slotsType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(slotsType),
        };
    }

    public IValue Visit(IEvaluateContext context, UpdatePagedAttentionKVCache target)
    {
        var slots = context.GetArgumentValueAsTensor(target, UpdatePagedAttentionKVCache.Slots);
        var kvCache = context.GetArgumentValue(target, UpdatePagedAttentionKVCache.KVCache);
        UpdateCache(slots, kvCache.AsObjectRef<PagedAttentionKVCache>(), target.CacheKind, target.LayerId);
        return kvCache;
    }

    private static void UpdateCache(Tensor slots, PagedAttentionKVCache cache, AttentionCacheKind cacheKind, int layerId)
    {
        if (slots.Dimensions[0] != cache.NumRequests)
        {
            throw new ArgumentException($"slots dim 0 {slots.Dimensions[0]} != cache num requests {cache.NumRequests}");
        }

        // [num_queries, num_heads, head_dim]
        long queryStart = 0;
        for (int requestId = 0; requestId < cache.NumRequests; requestId++)
        {
            var slotIds = cache.GetOutputSlotIds(cacheKind, layerId);
            var slotsCount = slotIds.Dimensions[0];
            for (int i = 0; i < slotsCount; i++)
            {
                var slotId = slotIds[i];
                var slot = IR.F.Tensors.GetItem(slots, queryStart++).Evaluate().AsTensor();
                cache.UpdateOutputSlot(cacheKind, slotId, slot);
            }
        }
    }

    private IRType Visit(ITypeInferenceContext context, UpdatePagedAttentionKVCache target, TensorType slots, IRType kvCache)
    {
        return kvCache;
    }

    private IRType Visit(ITypeInferenceContext context, UpdatePagedAttentionKVCache target, DistributedType slots, IRType kvCache)
    {
        return new InvalidType("not support distributed type");
    }
}
