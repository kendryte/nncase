// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.TIR.NTT;

public sealed partial class VectorizedBinary : NTTKernelOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(VectorizedBinary), 0, "lhs", ParameterKind.Input);

    /// <summary>
    /// Gets Other.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(VectorizedBinary), 1, "rhs", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(VectorizedBinary), 2, "output", ParameterKind.Input);

    public BinaryOp BinaryOp { get; }

    public IRArray<int> LhsVectorizedAxes { get; }

    public IRArray<Dimension> LhsPadedNums { get; }

    public IRArray<int> RhsVectorizedAxes { get; }

    public IRArray<Dimension> RhsPadedNums { get; }

    public override string DisplayProperty() => $"BinaryOp: {BinaryOp}, LhsVectorizedAxes: {LhsVectorizedAxes}, LhsPadedNums: {LhsPadedNums}, RhsVectorizedAxes: {RhsVectorizedAxes}, RhsPadedNums: {RhsPadedNums}";
}
