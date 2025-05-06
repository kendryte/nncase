// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.NN;

public abstract record RefAttentionKVCache(
    IAttentionConfig Config,
    int NumSeqs,
    int NumTokens,
    Tensor<long> ContextLens,
    Tensor<long> SeqLens) : IAttentionKVCache
{
    public long ContextLen(int requestId) => ContextLens[requestId];

    public long SeqLen(int requestId) => SeqLens[requestId];
}

public sealed record RefPagedAttentionKVCache(
    IAttentionConfig Config,
    int NumSeqs,
    int NumTokens,
    Tensor<long> ContextLens,
    Tensor<long> SeqLens,
    Tensor<long> BlockTable,
    Tensor<long> SlotMapping,
    int NumBlocks,
    Tensor KVCaches)
    : RefAttentionKVCache(
        Config,
        NumSeqs,
        NumTokens,
        ContextLens,
        SeqLens), IPagedAttentionKVCache
{
    IPagedAttentionConfig IPagedAttentionKVCache.Config => (IPagedAttentionConfig)Config;

    private IPagedAttentionConfig PagedAttentionConfig => (IPagedAttentionConfig)Config;

    public Tensor GetBlockId(int seqId, int contextId)
    {
        Debug.Assert(BlockTable.Rank == 3, "BlockTable must be 3D.");
        return BlockTable.View([seqId, contextId, 0], [1, 1, BlockTable.Dimensions[2]]).Squeeze(0, 1).AsContiguous();
    }

    public Tensor GetSlotId(int tokenId)
    {
        Debug.Assert(SlotMapping.Rank == 2, "SlotMapping must be 2D.");
        return SlotMapping.View([tokenId, 0], [1, SlotMapping.Dimensions[1]]).Squeeze(0).AsContiguous();
    }

    public Tensor GetBlock(AttentionCacheKind kind, int layerId, int headId, Tensor blockId)
    {
        var blockView = GetBlockViewFromStorage(kind, layerId, headId, blockId);
        return blockView.AsContiguous();
    }

    public void UpdateBlock(AttentionCacheKind kind, int layerId, int headId, Tensor blockId, Tensor block)
    {
        block.CopyTo(GetBlockViewFromStorage(kind, layerId, headId, blockId));
    }

    public Tensor GetSlot(AttentionCacheKind kind, int layerId, int headId, Tensor slotId)
    {
        return GetSlotViewFromStorage(kind, layerId, headId, slotId);
    }

    public void UpdateSlot(AttentionCacheKind kind, int layerId, int headId, Tensor slotId, Tensor slot)
    {
        var destView = GetSlotViewFromStorage(kind, layerId, headId, slotId);
        slot.CopyTo(destView);
    }

    public void UpdateSlots(AttentionCacheKind kind, int layerId, int headId, Tensor slots)
    {
        // slots : [num_tokens, numHeads, headDim ]
        for (int i = 0; i < NumTokens; i++)
        {
            var slotId = GetSlotId(i);
            var slot = slots.View([i, headId, 0], [1, 1, slots.Dimensions[2]]).Squeeze([0, 1]);
            UpdateSlot(kind, layerId, headId, slotId, slot);
        }
    }

    private Tensor GetSlotViewFromStorage(AttentionCacheKind kind, int layerId, int headId, Tensor slotId)
    {
        // slot_id : [len(topo) + 1].
        var slotIdValue = (long)slotId[slotId.Dimensions[^1] - 1];
        var blockIdValue = slotIdValue / PagedAttentionConfig.BlockSize;
        var blockOffset = slotIdValue % PagedAttentionConfig.BlockSize;

        var blockId = Tensor.Zeros<long>(slotId.Dimensions);
        slotId.AsContiguous().CopyTo(blockId);
        blockId[slotId.Dimensions[^1] - 1] = blockIdValue;

        var blockView = GetBlockViewFromStorage(kind, layerId, headId, blockId);
        var blockShape = blockView.Dimensions;
        var blockLayout = PagedAttentionConfig.BlockLayout;

        // PagedAttentionDimKind[] defaultLayout = [PagedAttentionDimKind.BlockSize, PagedAttentionDimKind.HeadDim];
        int[] defaultLayout = [-1, -1, -1, 0, -1, 1];
        long[] defaultStarts = [blockOffset, 0];
        long[] defaultShape = [1, PagedAttentionConfig.HeadDim];
        if (PagedAttentionConfig.PackedAxes.IndexOf(PagedAttentionDimKind.HeadDim) is int i && i != -1)
        {
            defaultShape[1] /= PagedAttentionConfig.Lanes[i];
        }

        var starts = blockLayout.Select(i => defaultStarts[defaultLayout[(int)i]]).ToArray();
        var shape = blockLayout.Select(i => defaultShape[defaultLayout[(int)i]]).ToArray();
        return blockView.View(starts, shape).Squeeze([blockLayout.IndexOf(PagedAttentionDimKind.BlockSize)]);
    }

    private Tensor GetBlockViewFromStorage(AttentionCacheKind kind, int layerId, int headId, Tensor blockId)
    {
        // blockId: [len(topo) + 1].
        var blockIdValue = (long)blockId[blockId.Dimensions[^1] - 1];
        PagedAttentionDimKind[] defaultLayout = [PagedAttentionDimKind.NumBlocks, PagedAttentionDimKind.NumLayers, PagedAttentionDimKind.KV, PagedAttentionDimKind.BlockSize, PagedAttentionDimKind.NumKVHeads, PagedAttentionDimKind.HeadDim];
        long[] defaultStarts = [blockIdValue, layerId, (long)kind, 0, (long)headId, 0];
        long[] defaultShape = [1, 1, 1, PagedAttentionConfig.BlockSize, 1, PagedAttentionConfig.HeadDim];
        if (PagedAttentionConfig.PackedAxes.IndexOf(PagedAttentionDimKind.HeadDim) is int i && i != -1)
        {
            defaultShape[^1] /= PagedAttentionConfig.Lanes[i];
        }

        var starts = PagedAttentionConfig.CacheLayout.Select(i => defaultStarts[(int)i]).ToArray();
        var shape = PagedAttentionConfig.CacheLayout.Select(i => defaultShape[(int)i]).ToArray();
        var squeeze_axes = Enumerable.Range(0, PagedAttentionConfig.CacheLayout.Count).Where(i => PagedAttentionConfig.CacheLayout[i] is not (PagedAttentionDimKind.BlockSize or PagedAttentionDimKind.HeadDim)).Select(i => i + PagedAttentionConfig.Topology.Count).ToArray();

        // process the topology.
        var topo_starts = Enumerable.Range(0, PagedAttentionConfig.Topology.Count).Select(i => (long)blockId[i]).ToArray();
        var topo_shape = Enumerable.Range(0, PagedAttentionConfig.Topology.Count).Select(i => 1L).ToArray();
        var topo_squeeze = Enumerable.Range(0, PagedAttentionConfig.Topology.Count).Select(i => i).ToArray();

        var final_starts = topo_starts.Concat(starts).ToArray();
        var final_shape = topo_shape.Concat(shape).ToArray();
        var final_squeeze = topo_squeeze.Concat(squeeze_axes).ToArray();
        return KVCaches.View(final_starts, final_shape).Squeeze(final_squeeze);
    }
}

public sealed class CreatePagedAttentionKVCacheEvaluator : ITypeInferencer<CreatePagedAttentionKVCache>, ICostEvaluator<CreatePagedAttentionKVCache>, IEvaluator<CreatePagedAttentionKVCache>
{
    public IRType Visit(ITypeInferenceContext context, CreatePagedAttentionKVCache target)
    {
        var num_seqs = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumSeqs);
        var num_tokens = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumTokens);
        var context_lens = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.ContextLens);
        var seq_lens = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.SeqLens);
        var block_table = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.BlockTable);
        var slot_mapping = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.SlotMapping);
        var num_blocks = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumBlocks);
        var kv_caches = context.CheckArgumentType<IRType>(target, CreatePagedAttentionKVCache.KvCaches);
        return (num_seqs, num_tokens, context_lens, seq_lens, block_table, slot_mapping, num_blocks, kv_caches) switch
        {
            (DistributedType dnum_seqs, DistributedType dnum_tokens, DistributedType dcontext_lens, DistributedType dseq_lens, DistributedType dblock_table, DistributedType dslot_mapping, DistributedType dnum_blocks, DistributedType dkv_caches) => VisitType(context, target, dnum_seqs, dnum_tokens, dcontext_lens, dseq_lens, dblock_table, dslot_mapping, dnum_blocks, dkv_caches),
            (TensorType tnum_seqs, TensorType tnum_tokens, TensorType tcontext_lens, TensorType tseq_lens, TensorType tblock_table, TensorType tslot_mapping, TensorType tnum_blocks, TensorType tkv_caches) => VisitType(context, target, tnum_seqs, tnum_tokens, tcontext_lens, tseq_lens, tblock_table, tslot_mapping, tnum_blocks, tkv_caches),
            _ => new InvalidType("not support type"),
        };
    }

    public Cost Visit(ICostEvaluateContext context, CreatePagedAttentionKVCache target)
    {
        var num_seqs = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumSeqs);
        var num_tokens = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumTokens);
        var context_lens = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.ContextLens);
        var seq_lens = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.SeqLens);
        var block_table = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.BlockTable);
        var slot_mapping = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.SlotMapping);
        var num_blocks = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.NumBlocks);
        var kv_caches = context.GetArgumentType<IRType>(target, CreatePagedAttentionKVCache.KvCaches);

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(num_seqs) + CostUtility.GetMemoryAccess(num_tokens) + CostUtility.GetMemoryAccess(context_lens) + CostUtility.GetMemoryAccess(seq_lens) + CostUtility.GetMemoryAccess(block_table) + CostUtility.GetMemoryAccess(slot_mapping) + CostUtility.GetMemoryAccess(num_blocks) + CostUtility.GetMemoryAccess(kv_caches),
        };
    }

    public IValue Visit(IEvaluateContext context, CreatePagedAttentionKVCache target)
    {
        var num_seqs = context.GetArgumentValueAsScalar<int>(target, CreatePagedAttentionKVCache.NumSeqs);
        var num_tokens = context.GetArgumentValueAsScalar<int>(target, CreatePagedAttentionKVCache.NumTokens);
        var context_lens = context.GetArgumentValueAsTensor<long>(target, CreatePagedAttentionKVCache.ContextLens);
        var seq_lens = context.GetArgumentValueAsTensor<long>(target, CreatePagedAttentionKVCache.SeqLens);
        var block_table = context.GetArgumentValueAsTensor<long>(target, CreatePagedAttentionKVCache.BlockTable);
        var slot_mapping = context.GetArgumentValueAsTensor<long>(target, CreatePagedAttentionKVCache.SlotMapping);
        var num_blocks = context.GetArgumentValueAsScalar<int>(target, CreatePagedAttentionKVCache.NumBlocks);
        var kv_caches = context.GetArgumentValueAsTensor(target, CreatePagedAttentionKVCache.KvCaches);

        var kv_cache = new RefPagedAttentionKVCache(target.Config, num_seqs, num_tokens, context_lens, seq_lens, block_table, slot_mapping, num_blocks, kv_caches);
        return Value.FromTensor(Tensor.FromScalar(new Reference<IPagedAttentionKVCache>(kv_cache)));
    }

    private IRType CheckAllBroadCast(DistributedType distributedType, [System.Runtime.CompilerServices.CallerArgumentExpression("distributedType")] string? name = null)
    {
        if (!distributedType.AxisPolices.All(x => x is SBPBroadCast))
        {
            return new InvalidType($"{name} is not all broadcast");
        }

        return distributedType;
    }

    private IRType VisitType(ITypeInferenceContext context, CreatePagedAttentionKVCache target, DistributedType num_seqs, DistributedType num_tokens, DistributedType context_lens, DistributedType seq_lens, DistributedType block_table, DistributedType slot_mapping, DistributedType num_blocks, DistributedType kv_caches)
    {
        var validType = VisitType(context, target, num_seqs.TensorType, num_tokens.TensorType, context_lens.TensorType, seq_lens.TensorType, block_table.TensorType, slot_mapping.TensorType, num_blocks.TensorType, kv_caches.TensorType);
        if (validType is InvalidType)
        {
            return validType;
        }

        if (CheckAllBroadCast(num_tokens) is InvalidType iv)
        {
            return iv;
        }

        if (CheckAllBroadCast(context_lens) is InvalidType iv1)
        {
            return iv1;
        }

        if (CheckAllBroadCast(seq_lens) is InvalidType iv2)
        {
            return iv2;
        }

        if (CheckAllBroadCast(block_table) is InvalidType iv3)
        {
            return iv3;
        }

        if (CheckAllBroadCast(slot_mapping) is InvalidType iv4)
        {
            return iv4;
        }

        if (CheckAllBroadCast(num_blocks) is InvalidType iv5)
        {
            return iv5;
        }

        if (kv_caches.Placement.Name == "cdxyt")
        {
            if (kv_caches.AxisPolices[0] is SBPSplit { Axes: [1] } &&
                kv_caches.AxisPolices[1] is SBPSplit { Axes: [2, 3] } &&
                kv_caches.AxisPolices.Skip(2).All(x => x is SBPBroadCast))
            {
                return validType;
            }
        }

        return new InvalidType("not support distributed kv caches");
    }

    private IRType VisitType(ITypeInferenceContext context, CreatePagedAttentionKVCache target, TensorType num_seqs, TensorType num_tokens, TensorType context_lens, TensorType seq_lens, TensorType block_table, TensorType slot_mapping, TensorType num_blocks, TensorType kv_caches)
    {
        if (!num_seqs.IsScalar)
        {
            return new InvalidType("num_seqs is not scalar");
        }

        if (!num_tokens.IsScalar)
        {
            return new InvalidType("num_tokens is not scalar");
        }

        if (context_lens.Shape.Rank != 1)
        {
            return new InvalidType("context_lens rank != 1");
        }

        if (seq_lens.Shape.Rank != 1)
        {
            return new InvalidType("seq_lens rank != 1");
        }

        if (block_table.Shape.Rank != 3)
        {
            return new InvalidType("block_table rank != 3");
        }

        if (slot_mapping.Shape.Rank != 2)
        {
            return new InvalidType("slot_mapping rank != 2");
        }

        if (!num_blocks.IsScalar)
        {
            return new InvalidType("slot_mapping rank != 2");
        }

        if (!num_blocks.IsScalar)
        {
            return new InvalidType("slot_mapping rank != 2");
        }

        if (kv_caches.Shape.Rank < target.Config.CacheLayout.Count)
        {
            return new InvalidType("kv_caches shape < CacheLayout.Count");
        }

        return new ReferenceType(new PagedAttentionKVCacheType()
        {
            Config = target.Config,
        });
    }
}
