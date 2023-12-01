// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class TensorLoad : CPUKernelOp
{
    public static readonly ParameterInfo Dest = new(typeof(TensorLoad), 0, "dest");

    public static readonly ParameterInfo Src = new(typeof(TensorLoad), 1, "src");

    public IRArray<SBP> NdSbp { get; }

    public Placement Placement { get; }
}
