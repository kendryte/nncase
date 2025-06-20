// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

[PatternFunctionalGenerator]
public sealed partial class PackedBinary : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(PackedBinary), 0, "lhs", ParameterKind.Input);

    /// <summary>
    /// Gets Other.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(PackedBinary), 1, "rhs", ParameterKind.Input);

    public BinaryOp BinaryOp { get; }

    public IRArray<int> LhsPackedAxes { get; }

    public IRArray<Dimension> LhsPadedNums { get; }

    public IRArray<int> RhsPackedAxes { get; }

    public IRArray<Dimension> RhsPadedNums { get; }

    public override string DisplayProperty() => $"BinaryOp: {BinaryOp}, LhsPackedAxes: {LhsPackedAxes}, LhsPadedNums: {LhsPadedNums}, RhsPackedAxes: {RhsPackedAxes}, RhsPadedNums: {RhsPadedNums}";
}
