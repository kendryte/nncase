// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Reflection.Metadata;
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class RoPE : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(RoPE), 0, "input");

    public static readonly ParameterInfo Cos = new(typeof(RoPE), 1, "cos");

    public static readonly ParameterInfo Sin = new(typeof(RoPE), 2, "sin");

    public static readonly ParameterInfo Output = new(typeof(RoPE), 3, "output");
}
