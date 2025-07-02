// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Math;
using Nncase.IR.NTT;
using Nncase.PatternMatch;
using Nncase.Targets;
using Nncase.TIR;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public static bool TryGetBinaryAffineRelation(IR.Shape lhsShape, IR.Shape rhsShape, out AffineDomain[] domains, out AffineMap lhsMap, out AffineMap rhsMap)
    {
        var rank = Math.Max(lhsShape.Rank, rhsShape.Rank);
        domains = IR.F.Affine.Domains(rank);
        lhsMap = null!;
        rhsMap = null!;
        var lhsRes = new AffineRange[lhsShape.Rank];
        var rhsRes = new AffineRange[rhsShape.Rank];
        for (int i = rank - 1; i >= 0; i--)
        {
            var lhsi = i - (rank - lhsShape.Rank);
            var rhsi = i - (rank - rhsShape.Rank);
#pragma warning disable SA1008 // Opening parenthesis should be spaced correctly
            switch (lhsi, rhsi)
            {
                case ( >= 0, >= 0):
                    switch (lhsShape[lhsi], rhsShape[rhsi])
                    {
                        case (DimConst constA, DimConst constB):
                            switch (constA.Value, constB.Value)
                            {
                                case (long a, long b) when a == b:
                                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    break;
                                case (1, _):
                                    lhsRes[lhsi] = new AffineRange(0, 1);
                                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    break;
                                case (_, 1):
                                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    rhsRes[rhsi] = new AffineRange(0, 1);
                                    break;
                                default:
                                    return false;
                            }

                            break;
                        case (DimConst constA, Dimension dimB):
                            switch (constA.Value, dimB.Metadata.Range)
                            {
                                case (1, { Min: >= 1 }):
                                    lhsRes[lhsi] = new AffineRange(0, 1);
                                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    break;
                                default:
                                    return false;
                            }

                            break;
                        case (Dimension dimA, DimConst constB):
                            switch (dimA.Metadata.Range, constB.Value)
                            {
                                case ({ Min: >= 1 }, 1):
                                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    rhsRes[rhsi] = new AffineRange(0, 1);
                                    break;
                                default:
                                    return false;
                            }

                            break;
                        case (Dimension dimA, Dimension dimB):
                            switch (dimA.Metadata.Range, dimB.Metadata.Range)
                            {
                                case (var rangeA, var rangeB) when rangeA == rangeB:
                                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                                    break;
                                default:
                                    return false;
                            }

                            break;
                        default:
                            return false;
                    }

                    break;
                case ( < 0, >= 0):
                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                    break;
                case ( >= 0, < 0):
                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                    break;
                case ( < 0, < 0):
                    break;
            }
#pragma warning restore SA1008 // Opening parenthesis should be spaced correctly
        }

        lhsMap = new AffineMap(domains, default, lhsRes);
        rhsMap = new AffineMap(domains, default, rhsRes);
        return true;
    }

    public Expr SelectPackedBinary(PackedBinary binary, Call call, Expr output)
    {
        var lhs = (Expr)call[PackedBinary.Lhs];
        var rhs = (Expr)call[PackedBinary.Rhs];
        if (lhs.CheckedShape is not { Rank: > 0 } || rhs.CheckedShape is not { Rank: > 0 })
        {
            return call;
        }

        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        if (!TryGetBinaryAffineRelation(lhsShape, rhsShape, out var domains, out var lhsMap, out var rhsMap))
        {
            return call;
        }

        return IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(lhs, lhsMap, out var lhsTile)
            .Read(rhs, rhsMap, out var rhsTile)
            .Write(output, AffineMap.Identity(domains.Length), out var outTile)
            .Body(TIR.F.NTT.PackedBinary(lhsTile, rhsTile, outTile, binary.BinaryOp, binary.LhsPackedAxes, binary.LhsPadedNums, binary.RhsPackedAxes, binary.RhsPadedNums))
            .Build();
    }
}
