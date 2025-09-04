// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Tensors;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public static bool TryGetWhereAffineRelation(IR.Shape condShape, IR.Shape lhsShape, IR.Shape rhsShape, [MaybeNullWhen(false)] out AffineDomain[] domains, [MaybeNullWhen(false)] out AffineMap condMap, [MaybeNullWhen(false)] out AffineMap lhsMap, [MaybeNullWhen(false)] out AffineMap rhsMap)
    {
        var rank = Math.Max(condShape.Rank, Math.Max(lhsShape.Rank, rhsShape.Rank));
        domains = IR.F.Affine.Domains(rank);
        condMap = null;
        lhsMap = null;
        rhsMap = null;
        var condRes = new AffineRange[condShape.Rank];
        var lhsRes = new AffineRange[lhsShape.Rank];
        var rhsRes = new AffineRange[rhsShape.Rank];
        var shapes = new[] { condShape, lhsShape, rhsShape };
        var ranges = new[] { condRes, lhsRes, rhsRes };
        for (int i = rank - 1; i >= 0; i--)
        {
            var condi = i - (rank - condShape.Rank);
            var lhsi = i - (rank - lhsShape.Rank);
            var rhsi = i - (rank - rhsShape.Rank);
            int[] axes = { condi, lhsi, rhsi };

            for (int j = 0; j < axes.Length; j++)
            {
                var axis = axes[j];
                var range = ranges[j];
                var shape = shapes[j];
                if (axis >= 0)
                {
                    switch (shape[axis])
                    {
                        case DimConst dimConst:
                            range[axis] = dimConst.Value == 1 ? new AffineRange(0, 1) : new AffineRange(domains[i].Offset, domains[i].Extent);
                            break;
                        case Dimension { Metadata.Range: { Min: >= 1 } }:
                            range[axis] = new AffineRange(domains[i].Offset, domains[i].Extent);
                            break;
                        default:
                            return false;
                    }
                }
            }
        }

        condMap = new AffineMap(domains, default, condRes);
        lhsMap = new AffineMap(domains, default, lhsRes);
        rhsMap = new AffineMap(domains, default, rhsRes);
        return true;
    }

    public static Expr SelectWhere(Where op, Call call, Expr output)
    {
        var cond = (Expr)call[Where.Cond];
        var x = (Expr)call[Where.X];
        var y = (Expr)call[Where.Y];
        if (cond.CheckedShape is not { Rank: > 0 } || x.CheckedShape is not { Rank: > 0 } || y.CheckedShape is not { Rank: > 0 })
        {
            return call;
        }

        if (!TryGetWhereAffineRelation(cond.CheckedShape, x.CheckedShape, y.CheckedShape, out var domains, out var condMap, out var lhsMap, out var rhsMap))
        {
            return call;
        }

        return IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(cond, condMap, out var condTile)
            .Read(x, lhsMap, out var lhsTile)
            .Read(y, rhsMap, out var rhsTile)
            .Write(output, AffineMap.Identity(domains.Length), out var outTile)
            .Body(TIR.F.NTT.Where(condTile, lhsTile, rhsTile, outTile))
            .Build();
    }
}
