// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.Targets;
using Nncase.TIR;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes;

public sealed partial class NTTAffineSelectionPass
{
    private Expr SelectBinary(Binary binary, Call call, Expr output)
    {
        var lhs = (Expr)call[Binary.Lhs];
        var rhs = (Expr)call[Binary.Rhs];
        if (lhs.CheckedShape is not { IsFixed: true, Rank: > 0 }
            || rhs.CheckedShape is not { IsFixed: true, Rank: > 0 })
        {
            return call;
        }

        // [8, 16] * [16]
        var lhsShape = lhs.CheckedShape.ToValueArray();
        var rhsShape = rhs.CheckedShape.ToValueArray();
        var rank = Math.Max(lhs.CheckedShape.Rank, rhs.CheckedShape.Rank);
        var domains = IR.F.Affine.Domains(rank);
        var lhsRes = new AffineRange[lhs.CheckedShape.Rank];
        var rhsRes = new AffineRange[rhs.CheckedShape.Rank];
        for (int i = rank - 1; i >= 0; i--)
        {
            var lhsi = i - (rank - lhs.CheckedShape.Rank);
            var rhsi = i - (rank - rhs.CheckedShape.Rank);
            switch (lhsi, rhsi)
            {
                case (>= 0, >= 0):
                    switch (lhsShape[lhsi], rhsShape[rhsi])
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
                            return call;
                    }

                    break;
                case (< 0, >= 0):
                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                    break;
                case (>= 0, < 0):
                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                    break;
                case (< 0, < 0):
                    break;
            }
        }

        var lhsMap = new AffineMap(domains, default, lhsRes);
        var rhsMap = new AffineMap(domains, default, rhsRes);
        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(lhs, lhsMap, out var lhsTile)
            .Read(rhs, rhsMap, out var rhsTile)
            .Write(output, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.NTT.Binary(binary.BinaryOp, lhsTile, rhsTile, outTile))
            .Build();
    }
}
