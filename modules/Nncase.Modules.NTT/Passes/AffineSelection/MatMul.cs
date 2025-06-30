// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectMatMul(Op op, Call call, Expr output)
    {
        var lhs = (Expr)call.Arguments[IR.Math.MatMul.Lhs.Index];
        var rhs = (Expr)call.Arguments[IR.Math.MatMul.Rhs.Index];

        // TODO: summa not support tiling for now.
        if ((lhs.CheckedType is DistributedType ldt && ldt.AxisPolices.Last() is SBPSplit)
            || output.CheckedShape is not { IsFixed: true, Rank: > 0 })
        {
            return call;
        }

        var lhsShape = lhs.CheckedShape.ToValueArray();
        var rhsShape = rhs.CheckedShape.ToValueArray();
        var rank = Math.Max(lhs.CheckedShape.Rank, rhs.CheckedShape.Rank) + 1;
        var domains = IR.F.Affine.Domains(rank);
        var lhsRes = new AffineRange[lhsShape.Length];
        var rhsRes = new AffineRange[rhsShape.Length];
        for (int i = rank - 1 - 3; i >= 0; i--)
        {
            var lhsi = i - (rank - (lhsShape.Length + 1));
            var rhsi = i - (rank - (rhsShape.Length + 1));
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

        var (om, ok, on) = (rank - 3, rank - 2, rank - 1);
        var (lm, lk) = (lhsShape.Length - 2, lhsShape.Length - 1);
        var (rk, rn) = (rhsShape.Length - 2, rhsShape.Length - 1);
        if (op is IR.NTT.PackedMatMul pm)
        {
            if (pm.TransposeA)
            {
                (lm, lk) = (lk, lm);
            }

            if (pm.TransposeB)
            {
                (rk, rn) = (rn, rk);
            }
        }

        lhsRes[lm] = new AffineRange(domains[om].Offset, domains[om].Extent);
        lhsRes[lk] = new AffineRange(domains[ok].Offset, domains[ok].Extent);
        rhsRes[rk] = lhsRes[lk];
        rhsRes[rn] = new AffineRange(domains[on].Offset, domains[on].Extent);

        var lhsMap = new AffineMap(domains, default, lhsRes);
        var rhsMap = new AffineMap(domains, default, rhsRes);
        var outMap = new AffineMap(domains, default, domains.SkipLast(3).Concat([domains[om], domains[on]]).Select(x => new AffineRange(x.Offset, x.Extent)).ToArray());
        return IR.F.Affine.Grid()
            .Domain(rank, out var domainVar)
            .Read(lhs, lhsMap, out var lhsTile)
            .Read(rhs, rhsMap, out var rhsTile)
            .Write(output, outMap, out var outTile)
            .Body(op switch
            {
                IR.Math.MatMul => TIR.F.NTT.Matmul(lhsTile, rhsTile, outTile, IR.F.Math.NotEqual(domainVar[ok][0], 0L)),
                IR.NTT.PackedMatMul pop => TIR.F.NTT.Matmul(lhsTile, rhsTile, outTile, IR.F.Math.NotEqual(domainVar[ok][0], 0L), pop.LhsPackedAxes, pop.RhsPackedAxes, pop.TransposeA, pop.TransposeB, pop.FusedReduce),
                _ => throw new System.Diagnostics.UnreachableException(),
            }).Build();
    }
}
