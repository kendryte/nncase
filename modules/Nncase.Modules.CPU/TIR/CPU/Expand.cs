// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class Expand : CPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Expand), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(Expand), 1, "output");

    public IRArray<int> Shape { get; }

    public IRArray<SBP> NdSbp { get; }
}
