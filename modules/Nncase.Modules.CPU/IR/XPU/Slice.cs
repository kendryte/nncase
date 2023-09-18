// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.XPU;

public sealed partial class Slice : XPUKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(Slice), 0, "input");

    public static readonly ParameterInfo Begins = new(typeof(Slice), 1, "begins");

    public static readonly ParameterInfo Ends = new(typeof(Slice), 2, "ends");

    public static readonly ParameterInfo Axes = new(typeof(Slice), 3, "axes");

    public DistributedType DistributedType { get; }
}
