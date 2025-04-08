// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// AttentionKVCache.
/// </summary>
public abstract class AttentionKVCache : IEquatable<AttentionKVCache>
{
    public bool Equals(AttentionKVCache? other) => other != null && ReferenceEquals(this, other);

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return obj is AttentionKVCache && Equals((AttentionKVCache)obj);
    }
}

/// <summary>
/// Prim type of <see cref="QuantParam"/>.
/// </summary>
public sealed record AttentionKVCacheType : ValueType
{
    /// <inheritdoc/>
    public override Type CLRType => typeof(AttentionKVCache);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => throw new NotSupportedException();

    /// <inheritdoc/>
    public override Guid Uuid { get; } = new("687ec623-1197-4684-83c8-38da36f8cfa5");

    /// <inheritdoc/>
    public override string ToString()
    {
        return "AttentionKVCacheType";
    }
}

public sealed record PagedAttentionKVCacheType : ValueType
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
        return "PagedAttentionKVCacheType";
    }
}
