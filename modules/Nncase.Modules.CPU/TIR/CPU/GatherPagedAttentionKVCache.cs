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

public sealed partial class GatherPagedAttentionKVCache : CPUKernelOp
{
    public static readonly ParameterInfo ShardId = new(typeof(GatherPagedAttentionKVCache), 0, "ShardId");

    public static readonly ParameterInfo KVCaches = new(typeof(GatherPagedAttentionKVCache), 1, "kvCaches");

    public static readonly ParameterInfo Output = new(typeof(GatherPagedAttentionKVCache), 2, "Output");
}
