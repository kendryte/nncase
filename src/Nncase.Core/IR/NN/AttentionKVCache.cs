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

public interface IAttentionConfig
{
    int NumLayers { get; }

    int NumKVHeads { get; }

    int HeadDim { get; }
}

/// <summary>
/// AttentionKVCache.
/// </summary>
public interface IAttentionKVCache
{
    /// <summary>
    /// Gets the config.
    /// </summary>
    AttentionConfig Config { get; }

    /// <summary>
    /// Gets the number of requests.
    /// </summary>
    int NumRequests { get; }

    /// <summary>
    /// Gets the context lens.
    /// </summary>
    /// <param name="requestId">The request id.</param>
    /// <returns>The context lens.</returns>
    /// <remarks>
    /// The context lens are used to identify the lengths of the blocks of key-value
    /// pairs that are used for the attention mechanism in the transformer model.
    /// </remarks>
    long GetContextLen(int requestId);

    /// <summary>
    /// Gets the sequence lens.
    /// </summary>
    /// <param name="requestId">The request id.</param>
    /// <returns>The sequence lens.</returns>
    /// <remarks>
    /// The sequence lens are used to identify the lengths of the sequences of
    /// key-value pairs that are used for the attention mechanism in the transformer model.
    /// </remarks>
    long GetSeqLen(int requestId);
}

public record AttentionConfig(int NumLayers, int NumKVHeads, int HeadDim) : IAttentionConfig;

/// <summary>
/// Prim type of <see cref="QuantParam"/>.
/// </summary>
public record AttentionKVCacheType : ValueType
{
    /// <inheritdoc/>
    public override Type CLRType => typeof(IAttentionKVCache);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => throw new NotSupportedException();

    /// <inheritdoc/>
    public override Guid Uuid { get; } = new("687ec623-1197-4684-83c8-38da36f8cfa5");

    /// <inheritdoc/>
    public override string ToString()
    {
        return "AttentionKVCache";
    }
}
