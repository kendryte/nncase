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
        var kvCache = context.CheckArgumentType<IRType>(target, UpdatePagedAttentionKVCache.KVCaches);
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
        var kvCaches = context.GetArgumentValue(target, UpdatePagedAttentionKVCache.KVCaches);
        UpdateCache(slots, kvCaches.AsTensor().Cast<Reference<IPagedAttentionKVCache>>(), target.CacheKind, target.LayerId);
        return kvCaches;
    }

    private static void UpdateCache(Tensor slots, Tensor<Reference<IPagedAttentionKVCache>> kvCaches, AttentionCacheKind cacheKind, int layerId)
    {
        // TODO: Support DP
        if (kvCaches.Length != 1)
        {
            throw new ArgumentException($"kvCaches length {kvCaches.Length} != 1");
        }

        var cache = kvCaches.Single().Value;

        if (cache.Config.Topology.Count > 0)
        {
            // only for xpu
            var (_, num_kv_head, _) = (slots.Dimensions[0], slots.Dimensions[1], slots.Dimensions[2]);
            if (cache.Config.Topology is [1, 2] && num_kv_head == cache.Config.NumKVHeads * 2)
            {
                for (int tok_id = 0; tok_id < cache.NumTokens; tok_id++)
                {
                    var slot_id = cache.GetSlotId(tok_id);
                    if (slot_id[0] is not -1L)
                    {
                        throw new InvalidOperationException("should be broadcast!");
                    }

                    for (int die_id = 0; die_id < 2; die_id++)
                    {
                        var die_slot_id = Tensor.Zeros(slot_id.ElementType, slot_id.Dimensions);
                        slot_id.CopyTo(die_slot_id);
                        die_slot_id[0] = die_id;

                        for (int headId = 0; headId < cache.Config.NumKVHeads; headId++)
                        {
                            var slot = slots.View([tok_id, (die_id * cache.Config.NumKVHeads) + headId, 0], [1, 1, slots.Dimensions[2]]).Squeeze(0, 1);
                            cache.UpdateSlot(cacheKind, layerId, headId, die_slot_id, slot);
                        }
                    }
                }
            }
            else
            {
                throw new NotSupportedException();
            }
        }
        else
        {
            // [num_tokens, slot_shape]
            for (int headId = 0; headId < cache.Config.NumKVHeads; headId++)
            {
                cache.UpdateSlots(cacheKind, layerId, headId, slots);
            }
        }
    }

    private IRType Visit(ITypeInferenceContext context, UpdatePagedAttentionKVCache target, TensorType slots, IRType kvCache)
    {
        return kvCache;
    }

    private IRType Visit(ITypeInferenceContext context, UpdatePagedAttentionKVCache target, DistributedType slots, IRType kvCache)
    {
        if (slots.Placement.Name == "cdxyt")
        {
            // for xpu.
            // seq split at x, head split at die and y
            if (slots.AxisPolices[0] is SBPSplit { Axes: [2] } &&
                slots.AxisPolices[1] is SBPSplit { Axes: [1, 3] } &&
                slots.AxisPolices[2] is SBPBroadCast)
            {
                return kvCache;
            }
        }

        return new InvalidType("not support distributed type");
    }
}
