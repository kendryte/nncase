// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.CPU;

[PatternFunctionalGenerator]
public sealed partial class PackedMatMul : PackedOp
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(PackedMatMul), 0, "lhs", ParameterKind.Input);

    /// <summary>
    /// Gets Other.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(PackedMatMul), 1, "rhs", ParameterKind.Input);

    [Flags]
    public enum PackKind : byte
    {
        None = 1 << 0,
        M = 1 << 1,
        K = 1 << 2,
        N = 1 << 3,
    }

    public IRArray<int> LhsPackedAxes { get; }

    public IRArray<int> LhsPadedNums { get; }

    public IRArray<int> RhsPackedAxes { get; }

    public IRArray<int> RhsPadedNums { get; }

    public bool TransposeA { get; }

    public bool TransposeB { get; }

    public static (PackKind Lhs, PackKind Rhs) GetPackKind(IRArray<int> lhsPackedAxes, IRArray<int> rhsPackedAxes)
    {
        switch (lhsPackedAxes.Count, rhsPackedAxes.Count)
        {
            case (0, 0):
                return (PackKind.None, PackKind.None);
            case (0, 1):
                return (PackKind.None, PackKind.N);
            case (1, 0):
                return (PackKind.M, PackKind.None);
            case (1, 1):
                return (PackKind.M, PackKind.N);
            case (1, 2):
                return (PackKind.K, PackKind.K | PackKind.N);
            case (2, 1):
                return (PackKind.M | PackKind.K, PackKind.K);
            case (2, 2):
                return (PackKind.M | PackKind.K, PackKind.K | PackKind.N);
            default:
                throw new NotSupportedException($"{lhsPackedAxes.Count}, {rhsPackedAxes.Count}");
        }
    }

    public static ((int LM, int LK) Lhs, (int RK, int RN) Rhs) GetAxes(PackedMatMul target, int[] lhs, int[] rhs)
    {
        var (lm, lk) = target.TransposeA ? (lhs.Rank - 1, lhs.Rank - 2) : (lhs.Rank - 2, lhs.Rank - 1);
        var (rk, rn) = target.TransposeB ? (rhs.Rank - 1, rhs.Rank - 2) : (rhs.Rank - 2, rhs.Rank - 1);
        return ((lm, lk), (rk, rn));
    }

    public override string DisplayProperty() => $"LhsPackedAxes: {LhsPackedAxes}, LhsPadedNums: {LhsPadedNums}, RhsPackedAxes: {RhsPackedAxes}, RhsPadedNums: {RhsPadedNums}, TransposeA: {TransposeA}, TransposeB: {TransposeB}";
}
