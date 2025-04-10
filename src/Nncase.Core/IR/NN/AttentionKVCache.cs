// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.NN;

public enum AttentionCacheKind
{
    Key,
    Value,
}

public record AttentionConfig(int NumLayers, int NumKVHeads, int HeadDim);

/// <summary>
/// AttentionKVCache.
/// </summary>
public abstract record AttentionKVCache(
    AttentionConfig Config,
    int NumDecodeTokens,
    int NumPrefillTokens,
    int NumRequests,
    int NumPrefills,
    Tensor<long> ContextLens,
    Tensor<long> SeqLens);

/// <summary>
/// Prim type of <see cref="QuantParam"/>.
/// </summary>
public record AttentionKVCacheType : ValueType
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
        return nameof(AttentionKVCache);
    }
}
