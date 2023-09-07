// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.


namespace Nncase.IR.XPU;

public sealed partial class TDMAStore : XPUKernelOp
{
    public static readonly ParameterInfo Src = new(typeof(TDMAStore), 0, "src");

    public static readonly ParameterInfo Dest = new(typeof(TDMAStore), 1, "dest");

    public IRArray<SBP> NdSbp { get; }

    public Placement Placement { get; }
}
