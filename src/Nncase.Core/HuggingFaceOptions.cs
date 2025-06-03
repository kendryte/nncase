// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase;

public enum HuggingFaceAttentionBackendKind : byte
{
    Default = 0,
    PagedAttention = 1,
}

public record HuggingFaceOptions
{
    public bool OutputAttentions { get; set; }

    public bool OutputHiddenStates { get; set; }

    public bool UseCache { get; set; }

    public HuggingFaceAttentionBackendKind AttenionBackend { get; set; }

    public IAttentionConfig Config { get; set; } = null!;

    public int MaxModelLen { get; set; } = 512;

    public static HuggingFaceOptions Default => new();
}
