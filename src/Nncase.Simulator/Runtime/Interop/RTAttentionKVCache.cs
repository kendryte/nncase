// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.Runtime.Interop;

/// <summary>
/// the Runtime <see cref="IAttentionConfig"/>.
/// </summary>
public class RTAttentionConfig : RTObject, IAttentionConfig
{
    internal RTAttentionConfig()
        : base(IntPtr.Zero)
    {
    }

    internal RTAttentionConfig(IntPtr handle, bool addRef = false)
        : base(handle)
    {
        if (addRef)
        {
            Native.ObjectAddRef(handle);
        }
    }

    public int NumLayers
    {
        get
        {
            Native.AttentionConfigGetNumLayers(this, out int numLayers).ThrowIfFailed();
            return numLayers;
        }

        set
        {
            Native.AttentionConfigSetNumLayers(this, value).ThrowIfFailed();
        }
    }

    public int NumKVHeads
    {
        get
        {
            Native.AttentionConfigGetNumKvHeads(this, out int numKvHeads).ThrowIfFailed();
            return numKvHeads;
        }

        set
        {
            Native.AttentionConfigSetNumKvHeads(this, value).ThrowIfFailed();
        }
    }

    public int HeadDim
    {
        get
        {
            Native.AttentionConfigGetHeadDim(this, out int headDim).ThrowIfFailed();
            return headDim;
        }

        set
        {
            Native.AttentionConfigSetHeadDim(this, value).ThrowIfFailed();
        }
    }

    public PrimType KVPrimType
    {
        get
        {
            Native.AttentionConfigGetKvType(this, out TypeCode kvType).ThrowIfFailed();
            return DataType.FromTypeCode(kvType);
        }

        set
        {
            Native.AttentionConfigSetKvType(this, value.TypeCode).ThrowIfFailed();
        }
    }

    /// <summary>
    /// convert <see cref="AttentionConfig"/> Value To <see cref="RTAttentionConfig"/>.
    /// </summary>
    public static RTAttentionConfig FromConfig(AttentionConfig cfg)
    {
        if (cfg is PagedAttentionConfig pagedCfg)
        {
            return RTPagedAttentionConfig.FromConfig(pagedCfg);
        }
        else if (cfg.GetType() == typeof(AttentionConfig))
        {
            Native.AttentionConfigCreate(cfg.NumLayers, cfg.NumKVHeads, cfg.HeadDim, cfg.KVPrimType.TypeCode, out var rtcfg).ThrowIfFailed();
            return rtcfg;
        }
        else
        {
            throw new NotSupportedException($"Unsupported config type: {cfg.GetType()}");
        }
    }

    public static RTAttentionConfig FromHandle(IntPtr handle, bool addRef = false)
    {
        try
        {
            return new RTAttentionConfig(handle, addRef);
        }
        catch
        {
            Native.ObjectRelease(handle);
            throw;
        }
    }
}

public sealed class RTPagedAttentionConfig : RTAttentionConfig, IPagedAttentionConfig
{
    internal RTPagedAttentionConfig()
        : base(IntPtr.Zero)
    {
    }

    internal RTPagedAttentionConfig(IntPtr handle, bool addRef = false)
        : base(handle, addRef)
    {
    }

    public int BlockSize
    {
        get
        {
            Native.PagedAttentionConfigGetBlockSize(this, out var blockSize).ThrowIfFailed();
            return blockSize;
        }

        set
        {
            Native.PagedAttentionConfigSetBlockSize(this, value).ThrowIfFailed();
        }
    }

    public IRArray<PagedKVCacheDimKind> CacheLayout
    {
        get
        {
            var layout = new PagedKVCacheDimKind[6];
            Native.PagedAttentionConfigGetCacheLayout(this, layout, layout.Length).ThrowIfFailed();
            return layout;
        }

        set
        {
            if (value.Count != 6)
            {
                throw new ArgumentException("Cache layout must have 6 dimensions");
            }

            var arr = value.ToArray();
            Native.PagedAttentionConfigSetCacheLayout(this, arr, arr.Length).ThrowIfFailed();
        }
    }

    public IRArray<PagedKVCacheDimKind> PackedAxes
    {
        get
        {
            var packedAxes = new PagedKVCacheDimKind[8]; // Using max size from small_vector
            int length = packedAxes.Length;
            Native.PagedAttentionConfigGetPackedAxes(this, packedAxes, length).ThrowIfFailed();
            return packedAxes.Where(x => Enum.IsDefined(x)).ToArray();
        }

        set
        {
            var arr = value.ToArray();
            Native.PagedAttentionConfigSetPackedAxes(this, arr, arr.Length).ThrowIfFailed();
        }
    }

    public IRArray<int> Lanes
    {
        get
        {
            var lanes = new int[8]; // Using a reasonable initial size
            int length = lanes.Length;
            Native.PagedAttentionConfigGetLanes(this, lanes, length).ThrowIfFailed();
            return lanes.Where(x => x != -1).ToArray();
        }

        set
        {
            var arr = value.ToArray();
            Native.PagedAttentionConfigSetLanes(this, arr, arr.Length).ThrowIfFailed();
        }
    }

    public IRArray<PagedKVCacheDimKind> ShardingAxes
    {
        get
        {
            var axes = new PagedKVCacheDimKind[8];
            int length = axes.Length;
            Native.PagedAttentionConfigGetShardingAxes(this, axes, length).ThrowIfFailed();
            return axes.Where(x => Enum.IsDefined(x)).ToArray();
        }

        set
        {
            var arr = value.ToArray();
            Native.PagedAttentionConfigSetShardingAxes(this, arr, arr.Length).ThrowIfFailed();
        }
    }

    public IRArray<SBPSplit> AxisPolicies
    {
        get
        {
            var policies = new List<SBPSplit>();

            for (int i = 0; i < ShardingAxes.Count; i++)
            {
                Native.PagedAttentionConfigGetAxisPolicyLen(this, i, out var len).ThrowIfFailed();

                var policy = new int[len];
                Native.PagedAttentionConfigGetAxisPolicy(this, i, policy, len).ThrowIfFailed();

                policies.Add(SBP.S(policy));
            }

            return policies.ToArray();
        }

        set
        {
            for (int i = 0; i < value.Count; i++)
            {
                var policy = value[i];
                Native.PagedAttentionConfigSetAxisPolicy(
                    this,
                    i,
                    policy.Axes.ToArray(),
                    policy.Axes.Count).ThrowIfFailed();
            }
        }
    }

    public static RTPagedAttentionConfig FromConfig(IPagedAttentionConfig cfg)
    {
        // 1. Flatten axis policies into arrays
        var policyLengths = new List<int>();
        var flattenedPolicies = new List<int>();
        foreach (var policy in cfg.AxisPolicies)
        {
            policyLengths.Add(policy.Axes.Count);
            flattenedPolicies.AddRange(policy.Axes);
        }

        // 2. Create config
        Native.PagedAttentionConfigCreate(
            cfg.NumLayers,
            cfg.NumKVHeads,
            cfg.HeadDim,
            cfg.KVPrimType.TypeCode,
            cfg.BlockSize,
            cfg.CacheLayout.ToArray(),
            cfg.PackedAxes.ToArray(),
            cfg.PackedAxes.Count,
            cfg.Lanes.ToArray(),
            cfg.Lanes.Count,
            cfg.ShardingAxes.ToArray(),
            cfg.ShardingAxes.Count,
            flattenedPolicies.ToArray(),
            policyLengths.ToArray(),
            out var rtcfg).ThrowIfFailed();

        return rtcfg;
    }

    public static new RTPagedAttentionConfig FromHandle(IntPtr handle, bool addRef = false)
    {
        try
        {
            return new RTPagedAttentionConfig(handle, addRef);
        }
        catch
        {
            Native.ObjectRelease(handle);
            throw;
        }
    }
}

/// <summary>
/// the Runtime <see cref="IAttentionKVCache"/>.
/// </summary>
public abstract class RTAttentionKVCache : RTObject, IAttentionKVCache
{
    internal RTAttentionKVCache()
        : base(IntPtr.Zero)
    {
    }

    internal RTAttentionKVCache(IntPtr handle, bool addRef = false)
        : base(handle)
    {
        if (addRef)
        {
            Native.ObjectAddRef(handle);
        }
    }

    public IAttentionConfig Config
    {
        get
        {
            Native.AttentionKVCacheGetConfig(this, out var config).ThrowIfFailed();
            return config;
        }
    }

    public int NumSeqs
    {
        get
        {
            Native.AttentionKVCacheGetNumSeqs(this, out var numSeqs).ThrowIfFailed();
            return numSeqs;
        }

        set
        {
            Native.AttentionKVCacheSetNumSeqs(this, value).ThrowIfFailed();
        }
    }

    public int NumTokens
    {
        get
        {
            Native.AttentionKVCacheGetNumTokens(this, out var numTokens).ThrowIfFailed();
            return numTokens;
        }

        set
        {
            Native.AttentionKVCacheSetNumTokens(this, value).ThrowIfFailed();
        }
    }

    public long ContextLen(int seqId)
    {
        throw new NotImplementedException();
    }

    public long SeqLen(int seqId)
    {
        throw new NotImplementedException();
    }
}

/// <summary>
/// the Runtime <see cref="IPagedAttentionKVCache"/>.
/// </summary>
public class RTPagedAttentionKVCache : RTAttentionKVCache, IPagedAttentionKVCache
{
    internal RTPagedAttentionKVCache()
        : base(IntPtr.Zero)
    {
    }

    internal RTPagedAttentionKVCache(IntPtr handle, bool addRef = false)
        : base(handle, addRef)
    {
        if (addRef)
        {
            Native.ObjectAddRef(handle);
        }
    }

    public int NumBlocks
    {
        get
        {
            Native.PagedAttentionKVCacheGetNumBlocks(this, out var numBlocks).ThrowIfFailed();
            return numBlocks;
        }
    }

    IPagedAttentionConfig IPagedAttentionKVCache.Config
    {
        get
        {
            Native.AttentionKVCacheGetConfig(this, out var config).ThrowIfFailed();
            return RTPagedAttentionConfig.FromHandle(config.DangerousGetHandle(), true);
        }
    }

    public static RTPagedAttentionKVCache FromHandle(IntPtr handle, bool addRef = false)
    {
        try
        {
            return new RTPagedAttentionKVCache(handle, addRef);
        }
        catch
        {
            Native.ObjectRelease(handle);
            throw;
        }
    }

    public static RTPagedAttentionKVCache Create(
        RTPagedAttentionConfig config,
        int num_seqs,
        int num_tokens,
        RTTensor context_lens,
        RTTensor seq_lens,
        RTTensor block_table,
        RTTensor slot_mapping,
        int num_blocks,
        int[] kv_shape)
    {
        Native.PagedAttentionKVCacheCreate(config, num_seqs, num_tokens, context_lens, seq_lens, block_table, slot_mapping, num_blocks, kv_shape, kv_shape.Length, out var handle).ThrowIfFailed();
        return handle;
    }

    public void SetKVCache(int[] indices, Tensor kv_cache)
    {
        Native.PagedAttentionKVCacheSetKVCache(this, indices, indices.Length, RTTensor.FromTensor(kv_cache)).ThrowIfFailed();
    }

    public Tensor GetBlock(AttentionCacheKind kind, int layerId, int headId, Tensor blockId)
    {
        throw new NotImplementedException();
    }

    public Tensor GetBlockId(int seqId, int contextId)
    {
        throw new NotImplementedException();
    }

    public Tensor GetSlot(AttentionCacheKind kind, int layerId, int headId, Tensor slotId)
    {
        throw new NotImplementedException();
    }

    public Tensor GetSlotId(int tokenId)
    {
        throw new NotImplementedException();
    }

    public void UpdateBlock(AttentionCacheKind kind, int layerId, int headId, Tensor blockId, Tensor block)
    {
        throw new NotImplementedException();
    }

    public void UpdateSlot(AttentionCacheKind kind, int layerId, int headId, Tensor slotId, Tensor slot)
    {
        throw new NotImplementedException();
    }

    public void UpdateSlots(AttentionCacheKind kind, int layerId, int headId, Tensor slots)
    {
        throw new NotImplementedException();
    }

    public long[] LogicalCacheDimensions() => throw new NotImplementedException();
}
