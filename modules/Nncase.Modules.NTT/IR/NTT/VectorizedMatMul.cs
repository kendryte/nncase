// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Math;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

[PatternFunctionalGenerator]
public sealed partial class VectorizedMatMul : Op
{
    /// <summary>
    /// Gets input.
    /// </summary>
    public static readonly ParameterInfo Lhs = new(typeof(VectorizedMatMul), 0, "lhs", ParameterKind.Input);

    /// <summary>
    /// Gets Other.
    /// </summary>
    public static readonly ParameterInfo Rhs = new(typeof(VectorizedMatMul), 1, "rhs", ParameterKind.Input);

    [Flags]
    public enum VectorizeKind : byte
    {
        None = 1 << 0,
        M = 1 << 1,
        K = 1 << 2,
        N = 1 << 3,
    }

    public DataType OutputDataType { get; }

    public IRArray<int> LhsVectorizedAxes { get; }

    public IRArray<int> RhsVectorizedAxes { get; }

    public bool TransposeA { get; }

    public bool TransposeB { get; }

    public bool FusedReduce { get; }

    public static ((int LM, int LK) Lhs, (int RK, int RN) Rhs) GetAxes(VectorizedMatMul target, int[] lhs, int[] rhs)
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

    public (VectorizeKind Lhs, VectorizeKind Rhs) GetVectorizeKind(int lhsRank, int rhsRank)
    {
        var dimInfo = GetDimInfo(lhsRank, rhsRank);
        switch (LhsVectorizedAxes.Count, RhsVectorizedAxes.Count)
        {
            case (0, 0):
                return (VectorizeKind.None, VectorizeKind.None);
            case (0, 1) when RhsVectorizedAxes[0] == dimInfo.Rn:
                return (VectorizeKind.None, VectorizeKind.N);
            case (1, 0) when LhsVectorizedAxes[0] == dimInfo.Lm:
                return (VectorizeKind.M, VectorizeKind.None);
            case (1, 1) when LhsVectorizedAxes[0] == dimInfo.Lm && RhsVectorizedAxes[0] == dimInfo.Rn:
                return (VectorizeKind.M, VectorizeKind.N);
            case (1, 1) when LhsVectorizedAxes[0] == dimInfo.Lk && RhsVectorizedAxes[0] == dimInfo.Rk:
                return (VectorizeKind.K, VectorizeKind.K);
            case (1, 2) when LhsVectorizedAxes[0] == dimInfo.Lk && RhsVectorizedAxes == [dimInfo.Rk, dimInfo.Rn]:
                return (VectorizeKind.K, VectorizeKind.K | VectorizeKind.N);
            case (2, 1) when LhsVectorizedAxes == [dimInfo.Lm, dimInfo.Lk] && RhsVectorizedAxes[0] == dimInfo.Rk:
                return (VectorizeKind.M | VectorizeKind.K, VectorizeKind.K);
            case (2, 2) when LhsVectorizedAxes == [dimInfo.Lm, dimInfo.Lk] && (RhsVectorizedAxes == [dimInfo.Rk, dimInfo.Rn] || RhsVectorizedAxes == [dimInfo.Rn, dimInfo.Rk]):
                return (VectorizeKind.M | VectorizeKind.K, VectorizeKind.K | VectorizeKind.N);
            default:
                throw new NotSupportedException($"{LhsVectorizedAxes.Count}, {RhsVectorizedAxes.Count}");
        }
    }

    public override string DisplayProperty() => $"LhsVectorizedAxes: {LhsVectorizedAxes}, RhsVectorizedAxes: {RhsVectorizedAxes}, TransposeA: {TransposeA}, TransposeB: {TransposeB}";
}
