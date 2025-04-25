// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.NN;

public enum PagedAttentionDimKind : int
{
    NumBlocks = 0,
    NumLayers,
    KV,
    BlockSize,
    NumKVHeads,
    HeadDim,
}

public interface IPagedAttentionConfig : IAttentionConfig
{
    int BlockSize { get; }

    IRArray<PagedAttentionDimKind> CacheLayout { get; }

    IRArray<PagedAttentionDimKind> BlockLayout => CacheLayout.Where(x => x is PagedAttentionDimKind.BlockSize or PagedAttentionDimKind.HeadDim).ToArray();

    IRArray<PagedAttentionDimKind> PackedAxes { get; }

    IRArray<int> Lanes { get; }

    IRArray<int> Topology { get; }
}

/// <summary>
/// kv cache layout: [num_blocks, num_layers, num_head, 2, block_size].
///    block layout: [num_head, block_size].
///     slot layout: [num_head].
/// note the slot or block may have different pack shape.
/// </summary>
public interface IPagedAttentionKVCache : IAttentionKVCache
{
    /// <summary>
    /// Gets the config.
    /// </summary>
    new IPagedAttentionConfig Config { get; }

    int NumBlocks { get; }

    /// <summary>
    /// Gets the context block ids.
    /// </summary>
    /// <param name="seqId">The seq id.</param>
    /// <param name="contextId">The context id.</param>
    /// <returns>The context block ids.</returns>
    /// <remarks>
    /// The context block ids are used to identify the blocks of key-value pairs
    /// that are used for the attention mechanism in the transformer model.
    /// </remarks>
    Tensor GetBlockId(int seqId, int contextId);

    /// <summary>
    /// Gets the output slot ids.
    /// </summary>
    /// <param name="tokenId">The token id.</param>
    /// <returns>The output slot ids.</returns>
    /// <remarks>
    /// The output slot ids are used to identify the slots of key-value pairs
    /// that are used for the attention mechanism in the transformer model.
    /// The kind parameter indicates whether the output slot is for keys or values.
    /// </remarks>
    Tensor GetSlotId(int tokenId);

    /// <summary>
    /// Gets the block.
    /// </summary>
    /// <param name="kind">The kind of the block.</param>
    /// <param name="layerId">The layer id.</param>
    /// <param name="headId">The head id.</param>
    /// <param name="blockId">The block id.</param>
    /// <returns>The block contains block size and head dim.</returns>
    Tensor GetBlock(AttentionCacheKind kind, int layerId, int headId, Tensor blockId);

    /// <summary>
    /// Updates the output slot.
    /// </summary>
    /// <param name="kind">The kind of the output slot.</param>
    /// <param name="layerId">The layer id.</param>
    /// <param name="headId"> The head id.</param>
    /// <param name="blockId"> The block id.</param>
    /// <param name="block"> the block tensor.</param>
    void UpdateBlock(AttentionCacheKind kind, int layerId, int headId, Tensor blockId, Tensor block);

    /// <summary>
    /// Gets the slot from kv cache.
    /// </summary>
    /// <param name="kind">The kind of the slot.</param>
    /// <param name="layerId">The layer id.</param>
    /// <param name="headId">The head Id.</param>
    /// <param name="slotId">The slot id.</param>
    /// <returns>The slot.</returns>
    Tensor GetSlot(AttentionCacheKind kind, int layerId, int headId, Tensor slotId);

    /// <summary>
    /// Updates the slot in the kv cache.
    /// </summary>
    /// <param name="kind">The kind of the output slot.</param>
    /// <param name="layerId">The layer id.</param>
    /// <param name="headId">The head Id.</param>
    /// <param name="slotId">The slot id.</param>
    /// <param name="slot">The slot.</param>
    void UpdateSlot(AttentionCacheKind kind, int layerId, int headId, Tensor slotId, Tensor slot);

    /// <summary>
    /// Updates the slots in the kv cache.
    /// the slots is batch of slot.
    /// </summary>
    /// <param name="kind">The kind of the output slot.</param>
    /// <param name="layerId">The layer id.</param>
    /// <param name="headId">The head Id.</param>
    /// <param name="slots">The slots.</param>
    void UpdateSlots(AttentionCacheKind kind, int layerId, int headId, Tensor slots);
}

public sealed record PagedAttentionConfig(int NumLayers, int NumKVHeads, int HeadDim, PrimType KVType, int BlockSize, IRArray<PagedAttentionDimKind> CacheLayout, IRArray<PagedAttentionDimKind> PackedAxes, IRArray<int> Lanes, IRArray<int> Topology)
    : AttentionConfig(NumLayers, NumKVHeads, HeadDim, KVType), IPagedAttentionConfig
{
}

public sealed record PagedAttentionKVCacheType() : AttentionKVCacheType
{
    public IPagedAttentionConfig Config { get; set; }

    /// <inheritdoc/>
    public override Type CLRType => typeof(IPagedAttentionKVCache);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => throw new NotSupportedException();

    /// <inheritdoc/>
    public override Guid Uuid { get; } = new("f6955016-f185-46fb-aa5f-fdcea1c89ef6");

    /// <inheritdoc/>
    public override string ToString()
    {
        return "PagedAttentionKVCache";
    }
}
