// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.XPU;

public sealed partial class Gather : XPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Gather), 0, "input");

    public static readonly ParameterInfo Indices = new(typeof(Gather), 1, "indices");

    public static readonly ParameterInfo Output = new(typeof(Gather), 2, "output");

    public int Axis { get; }
}
