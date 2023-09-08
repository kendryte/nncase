// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.XPU;


public sealed partial class BlockMMA : XPUKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(BlockMMA), 0, "input");

    public static readonly ParameterInfo Rhs = new(typeof(BlockMMA), 1, "input");

    public static readonly ParameterInfo Output = new(typeof(BlockMMA), 2, "output");

}
