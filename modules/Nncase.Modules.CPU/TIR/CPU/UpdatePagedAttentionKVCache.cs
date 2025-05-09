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

public sealed partial class UpdatePagedAttentionKVCache : CPUKernelOp
{
    public static readonly ParameterInfo Slots = new(typeof(UpdatePagedAttentionKVCache), 0, "slots");

    public static readonly ParameterInfo KVCaches = new(typeof(UpdatePagedAttentionKVCache), 1, "kvCaches");

    public AttentionCacheKind CacheKind { get; }

    public int LayerId { get; }
}
