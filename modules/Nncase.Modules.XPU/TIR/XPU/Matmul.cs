// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.XPU;

public sealed partial class Matmul : XPUKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(Matmul), 0, "lhs");

    public static readonly ParameterInfo Rhs = new(typeof(Matmul), 1, "rhs");

    public static readonly ParameterInfo Output = new(typeof(Matmul), 2, "output");
}
