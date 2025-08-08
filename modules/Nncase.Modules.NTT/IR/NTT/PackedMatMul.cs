// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Math;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

[PatternFunctionalGenerator]
public sealed partial class PackedMatMul : Op
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

    public DataType OutputDataType { get; }

    public IRArray<int> LhsPackedAxes { get; }

    public IRArray<int> RhsPackedAxes { get; }

    public bool TransposeA { get; }

    public bool TransposeB { get; }

    public bool FusedReduce { get; }

    public static ((int LM, int LK) Lhs, (int RK, int RN) Rhs) GetAxes(PackedMatMul target, int[] lhs, int[] rhs)
    {
        var (lm, lk) = target.TransposeA ? (lhs.Rank - 1, lhs.Rank - 2) : (lhs.Rank - 2, lhs.Rank - 1);
        var (rk, rn) = target.TransposeB ? (rhs.Rank - 1, rhs.Rank - 2) : (rhs.Rank - 2, rhs.Rank - 1);
        return ((lm, lk), (rk, rn));
    }

    public static MatMulDimInfo GetDimInfo(bool transposeA, bool transposeB, int lhsRank, int rhsRank)
    {
        var lhsM = transposeA ? lhsRank - 1 : lhsRank - 2;
        var lhsK = transposeA ? lhsRank - 2 : lhsRank - 1;
        var rhsK = transposeB ? rhsRank - 1 : rhsRank - 2;
        var rhsN = transposeB ? rhsRank - 2 : rhsRank - 1;
        return new MatMulDimInfo(lhsM, lhsK, rhsK, rhsN);
    }

    public MatMulDimInfo GetDimInfo(int lhsRank, int rhsRank) => GetDimInfo(TransposeA, TransposeB, lhsRank, rhsRank);

    public (PackKind Lhs, PackKind Rhs) GetPackKind(int lhsRank, int rhsRank)
    {
        var dimInfo = GetDimInfo(lhsRank, rhsRank);
        switch (LhsPackedAxes.Count, RhsPackedAxes.Count)
        {
            case (0, 0):
                return (PackKind.None, PackKind.None);
            case (0, 1) when RhsPackedAxes[0] == dimInfo.Rn:
                return (PackKind.None, PackKind.N);
            case (1, 0) when LhsPackedAxes[0] == dimInfo.Lm:
                return (PackKind.M, PackKind.None);
            case (1, 1) when LhsPackedAxes[0] == dimInfo.Lm && RhsPackedAxes[0] == dimInfo.Rn:
                return (PackKind.M, PackKind.N);
            case (1, 1) when LhsPackedAxes[0] == dimInfo.Lk && RhsPackedAxes[0] == dimInfo.Rk:
                return (PackKind.K, PackKind.K);
            case (1, 2) when LhsPackedAxes[0] == dimInfo.Lk && (RhsPackedAxes == [dimInfo.Rk, dimInfo.Rn] || RhsPackedAxes == [dimInfo.Rn, dimInfo.Rk]):
                return (PackKind.K, PackKind.K | PackKind.N);
            case (2, 1) when LhsPackedAxes == [dimInfo.Lm, dimInfo.Lk] && RhsPackedAxes[0] == dimInfo.Rk:
                return (PackKind.M | PackKind.K, PackKind.K);
            case (2, 2) when LhsPackedAxes == [dimInfo.Lm, dimInfo.Lk] && (RhsPackedAxes == [dimInfo.Rk, dimInfo.Rn] || RhsPackedAxes == [dimInfo.Rn, dimInfo.Rk]):
                return (PackKind.M | PackKind.K, PackKind.K | PackKind.N);
            default:
                throw new NotSupportedException($"{LhsPackedAxes.Count}, {RhsPackedAxes.Count}");
        }
    }

    public override string DisplayProperty() => $"LhsPackedAxes: {LhsPackedAxes}, RhsPackedAxes: {RhsPackedAxes}, TransposeA: {TransposeA}, TransposeB: {TransposeB}";
}
