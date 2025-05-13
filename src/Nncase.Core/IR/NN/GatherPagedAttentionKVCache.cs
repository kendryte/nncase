// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.PatternMatch;

namespace Nncase.IR.NN;

[PatternFunctionalGenerator]
public sealed partial class GatherPagedAttentionKVCache : Op
{
    public static readonly ParameterInfo ShardId = new(typeof(GatherPagedAttentionKVCache), 0, "ShardId", ParameterKind.Input);

    public static readonly ParameterInfo KVCaches = new(typeof(GatherPagedAttentionKVCache), 1, "kvCaches", ParameterKind.Attribute);

    public int NumBlocks { get; }
}
