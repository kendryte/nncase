// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;

namespace Nncase.TIR.CPU;

public sealed partial class PagedAttention : CPUKernelOp
{
    public static readonly ParameterInfo Q = new(typeof(PagedAttention), 0, "q");

    public static readonly ParameterInfo KVCaches = new(typeof(PagedAttention), 1, "kvCaches");

    public static readonly ParameterInfo Extra = new(typeof(PagedAttention), 2, "extra");

    public static readonly ParameterInfo Output = new(typeof(PagedAttention), 3, "Output");

    public int LayerId { get; }
}
