// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

public class PagedAttentionKVCache : AttentionKVCache, IEquatable<PagedAttentionKVCache>
{
    public PagedAttentionKVCache(int num_layers, int num_blocks, int block_size, int num_kv_heads, int head_size)
    {
        KCaches = new Tensor<float>([num_layers, num_blocks, block_size, num_kv_heads, head_size]);
        VCaches = new Tensor<float>([num_layers, num_blocks, block_size, num_kv_heads, head_size]);
        SeqLens = Tensor<long>.Empty;
        ContextLens = Tensor<long>.Empty;
        BlockTables = Tensor<long>.Empty;
        SlotMapping = Tensor<long>.Empty;
        NumLayers = num_layers;
        NumBlocks = num_blocks;
        BlockSize = block_size;
        NumKVHeads = num_kv_heads;
        HeadSize = head_size;
    }

    public Tensor<float> KCaches { get; }

    public Tensor<float> VCaches { get; }

    // [num_requests]
    public Tensor<long> SeqLens { get; set; }

    // [num_requests]
    public Tensor<long> ContextLens { get; set; }

    // [num_requests, max_num_blocks_per_seq]
    public Tensor<long> BlockTables { get; set; }

    // [num_tokens]
    public Tensor<long> SlotMapping { get; set; }

    public int NumLayers { get; }

    public int NumBlocks { get; }

    public int BlockSize { get; }

    public int NumKVHeads { get; }

    public int HeadSize { get; }

    public bool Equals(PagedAttentionKVCache? other)
    {
        return other is PagedAttentionKVCache cache &&
            KCaches.Equals(cache.KCaches) &&
            VCaches.Equals(cache.VCaches) &&
            SeqLens.Equals(cache.SeqLens) &&
            ContextLens.Equals(cache.ContextLens) &&
            BlockTables.Equals(cache.BlockTables) &&
            SlotMapping.Equals(cache.SlotMapping);
    }

    public override bool Equals(object? obj)
    {
        return Equals(obj as PagedAttentionKVCache);
    }
}
