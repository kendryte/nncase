// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.XPU;

public sealed partial class TDMALoad : XPUKernelOp
{
    public static readonly ParameterInfo Dest = new(typeof(TDMALoad), 0, "dest");

    public static readonly ParameterInfo Src = new(typeof(TDMALoad), 1, "src");

    public IRArray<SBP> NdSbp { get; }

    public Placement Placement { get; }
}
