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
public sealed partial class PagedAttention : Op
{
    public static readonly ParameterInfo Q = new(typeof(PagedAttention), 0, "q", ParameterKind.Input);

    public static readonly ParameterInfo K = new(typeof(PagedAttention), 1, "k", ParameterKind.Input);

    public static readonly ParameterInfo V = new(typeof(PagedAttention), 2, "v", ParameterKind.Input);

    public static readonly ParameterInfo KVCache = new(typeof(PagedAttention), 3, "kvcache", ParameterKind.Attribute);

    public long LayerId { get; }
}
