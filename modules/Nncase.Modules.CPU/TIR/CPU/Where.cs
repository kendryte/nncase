// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class Where : CPUKernelOp
{
    public static readonly ParameterInfo Cond = new(typeof(Where), 0, "cond");

    public static readonly ParameterInfo X = new(typeof(Where), 1, "x");

    public static readonly ParameterInfo Y = new(typeof(Where), 2, "y");

    public DistributedType DistType { get; }
}
