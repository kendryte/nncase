// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class Where : NTTKernelOp
{
    public static readonly ParameterInfo Cond = new(typeof(Where), 0, "cond");

    public static readonly ParameterInfo X = new(typeof(Where), 1, "x");

    public static readonly ParameterInfo Y = new(typeof(Where), 2, "y");
}
