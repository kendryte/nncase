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

    public int NumLayers => throw new NotImplementedException();

    public int NumKVHeads => throw new NotImplementedException();

    public int HeadDim => throw new NotImplementedException();

    /// <summary>
    /// convert <see cref="AttentionConfig"/> Value To <see cref="RTAttentionConfig"/>.
    /// </summary>
    public static RTAttentionConfig FromAttentionConfig(AttentionConfig value) => value switch
    {
        PagedAttentionConfig cfg => throw new NotImplementedException(),
        AttentionConfig cfg => throw new NotImplementedException(),
        _ => throw new ArgumentOutOfRangeException(nameof(value)),
    };

    public static RTAttentionConfig FromHandle(IntPtr handle, bool addRef = false)
    {
        try
        {
            throw new NotImplementedException();
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

    internal RTAttentionKVCache(IntPtr handle)
        : base(handle)
    {
    }

    public AttentionConfig Config => throw new NotImplementedException();

    public int NumRequests => throw new NotImplementedException();

    public static RTAttentionKVCache FromHandle(IntPtr handle, bool addRef = false)
    {
        try
        {
            throw new NotImplementedException();
        }
        catch
        {
            Native.ObjectRelease(handle);
            throw;
        }
    }

    public long GetContextLength(int requestId) => throw new NotImplementedException();

    public long GetSeqLens(int requestId) => throw new NotImplementedException();
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
        : base(handle)
    {
        if (addRef)
        {
            Native.ObjectAddRef(handle);
        }
    }

    public new PagedAttentionConfig Config => (PagedAttentionConfig)base.Config;

    public Tensor GetBlock(AttentionCacheKind kind, int layerId, object blockId) => throw new NotImplementedException();

    public Tensor GetContextBlockIds(int requestId) => throw new NotImplementedException();

    public Tensor GetOutputSlotIds() => throw new NotImplementedException();

    public Tensor GetSlot(AttentionCacheKind kind, int layerId, object slotId) => throw new NotImplementedException();

    public Tensor GetSlots(Tensor block, int startSlot, int count) => throw new NotImplementedException();

    public void UpdateOutputSlot(AttentionCacheKind kind, int layerId, object slotId, Tensor slot) => throw new NotImplementedException();
}
