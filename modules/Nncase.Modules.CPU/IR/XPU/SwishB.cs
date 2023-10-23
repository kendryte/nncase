// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.XPU;

public sealed partial class SwishB : XPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(SwishB), 0, "input");

    public static readonly ParameterInfo Output = new(typeof(SwishB), 1, "output");

    public float Beta { get; }
}