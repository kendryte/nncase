// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.CPU;

public sealed partial class SUMMA : CPUKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(SUMMA), 0, "lhs");

    public static readonly ParameterInfo Rhs = new(typeof(SUMMA), 1, "rhs");

    public static readonly ParameterInfo Output = new(typeof(SUMMA), 2, "output");

    public static readonly ParameterInfo LoadC = new(typeof(SUMMA), 3, "loadC");

    public IRArray<int> LhsPackedAxes { get; }

    public IRArray<int> LhsPadedNums { get; }

    public IRArray<int> RhsPackedAxes { get; }

    public IRArray<int> RhsPadedNums { get; }

    public bool TransposeA { get; }

    public bool TransposeB { get; }

    public override string DisplayProperty() => $"LhsPackedAxes: {LhsPackedAxes}, LhsPadedNums: {LhsPadedNums}, RhsPackedAxes: {RhsPackedAxes}, RhsPadedNums: {RhsPadedNums}, TransposeA: {TransposeA}, TransposeB: {TransposeB}";
}
