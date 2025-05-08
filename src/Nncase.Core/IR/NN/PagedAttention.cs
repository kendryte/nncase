﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.PatternMatch;

namespace Nncase.IR.NN;

public enum AttentionDimKind : int
{
    Seq = 0,
    Head,
    Dim,
}

[PatternFunctionalGenerator]
public sealed partial class PagedAttention : Op
{
    public static readonly ParameterInfo Q = new(typeof(PagedAttention), 0, "q", ParameterKind.Input);

    public static readonly ParameterInfo KVCaches = new(typeof(PagedAttention), 1, "kvCaches", ParameterKind.Attribute);

    public static readonly ParameterInfo Extra = new(typeof(PagedAttention), 2, "extra", ParameterKind.Input);

    public int LayerId { get; }

    public IRArray<AttentionDimKind> QLayout { get; }
}
