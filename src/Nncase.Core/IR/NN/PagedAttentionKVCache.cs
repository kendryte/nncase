// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.NN;

public interface IPagedAttentionConfig : IAttentionConfig
{
    int BlockSize { get; }
}

public interface IPagedAttentionKVCache : IAttentionKVCache
{
    /// <summary>
    /// Gets the config.
    /// </summary>
    new PagedAttentionConfig Config { get; }

    /// <summary>
    /// Gets the contiguous sub block.
    /// </summary>
    /// <param name="indices">indices.</param>
    /// <returns> block tensor. </returns>
    Tensor GetSubBlock(params int[] indices);

    /// <summary>
    /// Sets the contiguous sub block.
    /// </summary>
    /// <param name="indices">indices.</param>
    /// <param name="subBlock">block tensor.</param>
    void SetSubBlock(int[] indices, Tensor subBlock);

    /// <summary>
    /// Gets the context block ids.
    /// </summary>
    /// <param name="requestId">The request id.</param>
    /// <returns>The context block ids.</returns>
    /// <remarks>
    /// The context block ids are used to identify the blocks of key-value pairs
    /// that are used for the attention mechanism in the transformer model.
    /// </remarks>
    Tensor GetContextBlockIds(int requestId);

    /// <summary>
    /// Gets the block.
    /// </summary>
    /// <param name="kind">The kind of the block.</param>
    /// <param name="layerId">The layer id.</param>
    /// <param name="blockId">The block id.</param>
    /// <returns>The block.</returns>
    /// <remarks>
    /// The block is used to store the key-value pairs for the attention mechanism
    /// in the transformer model. The kind parameter indicates whether the block
    /// is for keys or values.
    /// </remarks>
    Tensor GetBlock(AttentionCacheKind kind, int layerId, object blockId);

    /// <summary>
    /// Gets the slots.
    /// </summary>
    /// <param name="block">The block.</param>
    /// <param name="startSlot">The start slot.</param>
    /// <param name="count">The count.</param>
    /// <returns>The slots.</returns>
    /// <remarks>
    /// The slots are used to store the key-value pairs for the attention mechanism
    /// in the transformer model. The block parameter indicates which block the
    /// slots belong to, and the startSlot and count parameters indicate which
    /// slots to retrieve.
    /// </remarks>
    Tensor GetSlots(Tensor block, int startSlot, int count);

    /// <summary>
    /// Gets the slot.
    /// </summary>
    /// <param name="kind">The kind of the slot.</param>
    /// <param name="layerId">The layer id.</param>
    /// <param name="slotId">The slot id.</param>
    /// <returns>The slot.</returns>
    /// <remarks>
    /// The slot is used to store the key-value pairs for the attention mechanism
    /// in the transformer model. The kind parameter indicates whether the slot
    /// is for keys or values.
    /// </remarks>
    Tensor GetSlot(AttentionCacheKind kind, int layerId, object slotId);

    /// <summary>
    /// Gets the output slot ids.
    /// </summary>
    /// <returns>The output slot ids.</returns>
    /// <remarks>
    /// The output slot ids are used to identify the slots of key-value pairs
    /// that are used for the attention mechanism in the transformer model.
    /// The kind parameter indicates whether the output slot is for keys or values.
    /// </remarks>
    Tensor GetOutputSlotIds();

    /// <summary>
    /// Updates the output slot.
    /// </summary>
    /// <param name="kind">The kind of the output slot.</param>
    /// <param name="layerId">The layer id.</param>
    /// <param name="slotId">The slot id.</param>
    /// <param name="slot">The slot.</param>
    /// <remarks>
    /// The output slot is used to store the key-value pairs for the attention
    /// mechanism in the transformer model. The kind parameter indicates whether
    /// the output slot is for keys or values. The slot parameter contains the
    /// updated key-value pairs.
    /// </remarks>
    void UpdateOutputSlot(AttentionCacheKind kind, int layerId, object slotId, Tensor slot);
}

public sealed record PagedAttentionConfig(int BlockSize, int NumLayers, int NumKVHeads, int HeadDim)
    : AttentionConfig(NumLayers, NumKVHeads, HeadDim), IPagedAttentionConfig;

public sealed record PagedAttentionKVCacheType : AttentionKVCacheType
{
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
