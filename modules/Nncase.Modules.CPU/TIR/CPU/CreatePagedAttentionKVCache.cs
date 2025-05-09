﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;

namespace Nncase.TIR.CPU;

public sealed partial class CreatePagedAttentionKVCache : CPUKernelOp
{
    public static readonly ParameterInfo NumSeqs = new(typeof(CreatePagedAttentionKVCache), 0, "num_seqs", ParameterKind.Input);

    public static readonly ParameterInfo NumTokens = new(typeof(CreatePagedAttentionKVCache), 1, "num_tokens", ParameterKind.Input);

    public static readonly ParameterInfo ContextLens = new(typeof(CreatePagedAttentionKVCache), 2, "context_lens", ParameterKind.Input);

    public static readonly ParameterInfo SeqLens = new(typeof(CreatePagedAttentionKVCache), 3, "seq_lens", ParameterKind.Input);

    public static readonly ParameterInfo BlockTable = new(typeof(CreatePagedAttentionKVCache), 4, "block_table", ParameterKind.Input);

    public static readonly ParameterInfo SlotMapping = new(typeof(CreatePagedAttentionKVCache), 5, "slot_mapping", ParameterKind.Input);

    public static readonly ParameterInfo NumBlocks = new(typeof(CreatePagedAttentionKVCache), 6, "num_blocks", ParameterKind.Input);

    public static readonly ParameterInfo KvCaches = new(typeof(CreatePagedAttentionKVCache), 7, "kv_caches", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(CreatePagedAttentionKVCache), 8, "output", ParameterKind.Input);

    public IPagedAttentionConfig Config { get; }
}
