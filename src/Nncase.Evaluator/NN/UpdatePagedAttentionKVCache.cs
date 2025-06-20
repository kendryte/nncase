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
        UpdateCache(slots, kvCaches.AsTensor().Cast<Reference<IPagedAttentionKVCache>>(), target.CacheKind, target.LayerId, target.Layout);
        return kvCaches;
    }

    private static void UpdateCache(Tensor slots, Tensor<Reference<IPagedAttentionKVCache>> kvCaches, AttentionCacheKind cacheKind, int layerId, IRArray<AttentionDimKind> layout)
    {
        // TODO: Support DP
        if (kvCaches.Length != 1)
        {
            throw new ArgumentException($"kvCaches length {kvCaches.Length} != 1");
        }

        var cache = kvCaches.Single().Value;

        var shape = slots.Dimensions.ToArray();
        shape[layout.IndexOf(AttentionDimKind.Seq)] = 1;
        shape[layout.IndexOf(AttentionDimKind.Head)] = 1;
        var starts = new long[shape.Length];
        var axes = new long[] { layout.IndexOf(AttentionDimKind.Seq), layout.IndexOf(AttentionDimKind.Head) };
        for (int tokenId = 0; tokenId < cache.NumTokens; tokenId++)
        {
            var slotId = cache.GetSlotId(tokenId);
            starts[layout.IndexOf(AttentionDimKind.Seq)] = tokenId;
            for (int headId = 0; headId < cache.Config.NumKVHeads; headId++)
            {
                starts[layout.IndexOf(AttentionDimKind.Head)] = headId;
                var slot = slots.View(starts, shape).Squeeze(axes); // only contains dim.

                // check the sharding axes.
                var headIdCopy = headId;
                var slotIdCopy = slotId.AsContiguous(true);
                for (int shardId = 0; shardId < cache.Config.ShardingAxes.Count; shardId++)
                {
                    switch (cache.Config.ShardingAxes[shardId])
                    {
                        case PagedKVCacheDimKind.NumKVHeads when slotIdCopy[shardId] is -1L:
                            var headTile = cache.Config.NumKVHeads / (int)cache.LogicalCacheDimensions()[shardId];
                            slotIdCopy[shardId] = System.Math.DivRem(headIdCopy, headTile, out headIdCopy);
                            break;
                        case PagedKVCacheDimKind.NumBlocks when slotIdCopy[shardId] is not -1L:
                            break;
                        default:
                            throw new ArgumentOutOfRangeException(nameof(slots));
                    }
                }

                cache.UpdateSlot(cacheKind, layerId, headIdCopy, slotIdCopy, slot);
            }
        }
    }

    private IRType Visit(ITypeInferenceContext context, UpdatePagedAttentionKVCache target, TensorType slots, IRType kvCache)
    {
        return kvCache;
    }

    private IRType Visit(ITypeInferenceContext context, UpdatePagedAttentionKVCache target, DistributedType slots, IRType kvCache)
    {
        // for xpu.
        if (slots.Placement.Name == "cdxyt")
        {
            if (!target.Layout.SequenceEqual([AttentionDimKind.Head, AttentionDimKind.Dim, AttentionDimKind.Seq]))
            {
                return new InvalidType("layout should be [head, dim, seq]");
            }

            // seq split at x, head split at die and y
            var seqAxis = target.Layout.IndexOf(AttentionDimKind.Seq);
            var headAxis = target.Layout.IndexOf(AttentionDimKind.Head);
            var dimAxis = target.Layout.IndexOf(AttentionDimKind.Dim);
            if (slots.AxisPolicies[seqAxis] is SBPSplit { Axes: [2] } &&
                slots.AxisPolicies[headAxis] is SBPSplit { Axes: [1, 3] } &&
                slots.AxisPolicies[dimAxis] is SBPBroadCast)
            {
                return kvCache;
            }
        }
        else if (slots.Placement.Hierarchy.SequenceEqual([1]))
        {
            return kvCache;
        }

        return new InvalidType("not support distributed type");
    }
}
