// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.IR.CustomNTT;

[PatternFunctionalGenerator]
public sealed partial class MatMul : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(MatMul), 0, "lhs", ParameterKind.Input);

    /// <summary>
    /// Gets Other.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(MatMul), 1, "rhs", ParameterKind.Input);

    public IRArray<int> LhsVectorizedAxes { get; }

    public IRArray<int> RhsVectorizedAxes { get; }

    public bool TransposeA { get; }

    public bool TransposeB { get; }

    public IRArray<SBP> LhsSBPs { get; }

    public IRArray<SBP> RhsSBPs { get; }

    public IRArray<SBP> OutSBPs { get; }

    public Cost Cost { get; }

    public string CSourcePath { get; }

    public string FuncName { get; }

    public override string DisplayProperty() => $"LhsVectorizedAxes: {LhsVectorizedAxes}, RhsVectorizedAxes: {RhsVectorizedAxes}, TransposeA: {TransposeA}, TransposeB: {TransposeB}";
}
