// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.NN;

public interface IPagedAttentionScheduler
{
    void Initialize(PagedAttentionConfig config, int numBlocks);

    PagedAttentionKVCache Schedule(Tensor<long> sessionIds, Tensor<long> tokensCount);
}

public sealed record PagedAttentionConfig(int BlockSize, int NumLayers, int NumKVHeads, int HeadDim)
    : AttentionConfig(NumLayers, NumKVHeads, HeadDim);

public abstract record PagedAttentionKVCache(PagedAttentionConfig Config,
    int NumDecodeTokens,
    int NumPrefillTokens,
    int NumRequests,
    int NumPrefills,
    Tensor<long> ContextLens,
    Tensor<long> SeqLens,
    Tensor BlockTables,
    Tensor SlotMapping,
    Tensor KVCaches)
    : AttentionKVCache(Config, NumDecodeTokens, NumPrefillTokens, NumRequests, NumPrefills, ContextLens, SeqLens)
{
    public new PagedAttentionConfig Config => (PagedAttentionConfig)base.Config;

    public abstract Tensor GetContextBlockIds(int requestId, int layerId);

    public abstract Tensor GetBlock(AttentionCacheKind kind, object blockId);

    public abstract Tensor GetSlots(Tensor block, int startSlot, int count);

    public abstract Tensor GetSlot(AttentionCacheKind kind, object slotId);

    public abstract Tensor GetOutputSlotIds(AttentionCacheKind kind, int layerId);

    public abstract Tensor GetOutputSlotIds(AttentionCacheKind kind, int requestId, int layerId);

    public abstract void UpdateOutputSlot(AttentionCacheKind kind, object slotId, Tensor slot);
}

public sealed record PagedAttentionKVCacheType : AttentionKVCacheType
{
    /// <inheritdoc/>
    public override Type CLRType => typeof(PagedAttentionKVCache);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => throw new NotSupportedException();

    /// <inheritdoc/>
    public override Guid Uuid { get; } = new("f6955016-f185-46fb-aa5f-fdcea1c89ef6");

    /// <inheritdoc/>
    public override string ToString()
    {
        return nameof(PagedAttentionKVCacheType);
    }
}
