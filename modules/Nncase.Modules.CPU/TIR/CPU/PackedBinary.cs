﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.TIR.CPU;

public sealed partial class PackedBinary : CPUKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(PackedBinary), 0, "lhs", ParameterKind.Input);

    /// <summary>
    /// Gets Other.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(PackedBinary), 1, "rhs", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(PackedBinary), 2, "output", ParameterKind.Input);

    public BinaryOp BinaryOp { get; }

    public IRArray<int> LhsPackedAxes { get; }

    public IRArray<int> LhsPadedNums { get; }

    public IRArray<int> RhsPackedAxes { get; }

    public IRArray<int> RhsPadedNums { get; }

    public override string DisplayProperty() => $"BinaryOp: {BinaryOp}, LhsPackedAxes: {LhsPackedAxes}, LhsPadedNums: {LhsPadedNums}, RhsPackedAxes: {RhsPackedAxes}, RhsPadedNums: {RhsPadedNums}";
}