// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class PackedMatMul : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedMatMul), 0, "lhs");

    public static readonly ParameterInfo Rhs = new(typeof(PackedMatMul), 1, "rhs");

    public static readonly ParameterInfo Output = new(typeof(PackedMatMul), 2, "output");

    public static readonly ParameterInfo LoadC = new(typeof(PackedMatMul), 3, "loadC");

    public bool FusedReduce { get; }

    public override string DisplayProperty() => $"FusedReduce: {FusedReduce}";
}
