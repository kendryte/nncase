// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class Matmul : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(Matmul), 0, "lhs");

    public static readonly ParameterInfo Rhs = new(typeof(Matmul), 1, "rhs");

    public static readonly ParameterInfo Output = new(typeof(Matmul), 2, "output");

    public static readonly ParameterInfo LoadC = new(typeof(Matmul), 3, "loadC");

    public IRArray<int> LhsVectorizedAxes { get; }

    public IRArray<int> RhsVectorizedAxes { get; }

    public bool TransposeA { get; }

    public bool TransposeB { get; }

    public bool FusedReduce { get; }

    public string CSourcePath { get; }

    public string FuncName { get; }

    public override string DisplayProperty() => $"LhsVectorizedAxes: {LhsVectorizedAxes}, RhsVectorizedAxes: {RhsVectorizedAxes}, TransposeA: {TransposeA}, TransposeB: {TransposeB}";
}
