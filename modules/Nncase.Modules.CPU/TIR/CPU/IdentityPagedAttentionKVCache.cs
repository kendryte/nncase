// Copyright (c) Canaan Inc. All rights reserved.
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

public sealed partial class IdentityPagedAttentionKVCache : CPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(IdentityPagedAttentionKVCache), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo NumSeqs = new(typeof(IdentityPagedAttentionKVCache), 1, "num_seqs", ParameterKind.Input);

    public static readonly ParameterInfo NumTokens = new(typeof(IdentityPagedAttentionKVCache), 2, "num_tokens", ParameterKind.Input);

    public static readonly ParameterInfo ContextLens = new(typeof(IdentityPagedAttentionKVCache), 3, "context_lens", ParameterKind.Input);

    public static readonly ParameterInfo SeqLens = new(typeof(IdentityPagedAttentionKVCache), 4, "seq_lens", ParameterKind.Input);

    public static readonly ParameterInfo BlockTable = new(typeof(IdentityPagedAttentionKVCache), 5, "block_table", ParameterKind.Input);

    public static readonly ParameterInfo SlotMapping = new(typeof(IdentityPagedAttentionKVCache), 6, "slot_mapping", ParameterKind.Input);

    public static readonly ParameterInfo NumBlocks = new(typeof(IdentityPagedAttentionKVCache), 7, "num_blocks", ParameterKind.Input);

    public static readonly ParameterInfo KvCaches = new(typeof(IdentityPagedAttentionKVCache), 8, "kv_caches", ParameterKind.Input);
}
