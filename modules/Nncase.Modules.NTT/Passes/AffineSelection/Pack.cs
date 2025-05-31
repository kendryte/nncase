// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;
using Pack = Nncase.IR.Tensors.Pack;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    private Expr SelectPack(Pack pack, Call call, Expr output)
    {
        var input = (Expr)call[Pack.Input];
        if (output.CheckedShape is not { IsFixed: true, Rank: > 0 })
        {
            return call;
        }

        var inputShape = input.CheckedShape.ToValueArray();
        var rank = inputShape.Length;
        var domains = IR.F.Affine.Domains(rank);
        var results = new AffineRange[rank];

        for (int axis = 0; axis < rank; axis++)
        {
            // e.g. f32[128,256] -> f32<4>[32,256]
            if (pack.Axes.IndexOf(axis) is int i && i != -1)
            {
                results[axis] = new AffineRange(pack.Lanes[i] * domains[axis].Offset, pack.Lanes[i] * domains[axis].Extent);
            }
            else
            {
                results[axis] = new AffineRange(domains[axis].Offset, domains[axis].Extent);
            }
        }

        var affinemap = new AffineMap(domains, default, results);
        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(input, affinemap, out var intile)
            .Write(output, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.NTT.Pack(intile, outTile, pack.Lanes, pack.Axes))
            .Build();
    }
}
