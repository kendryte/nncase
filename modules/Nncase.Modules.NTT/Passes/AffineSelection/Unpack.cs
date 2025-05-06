// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public sealed partial class NTTAffineSelectionPass
{
    private Expr SelectUnpack(IR.NTT.Unpack unpack, Call call, Expr output)
    {
        var input = (Expr)call[IR.NTT.Unpack.Input];
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
            if (unpack.Axes.IndexOf(axis) is int i && i != -1)
            {
                results[axis] = new AffineRange(unpack.Lanes[i] * domains[axis].Offset, unpack.Lanes[i] * domains[axis].Extent);
            }
            else
            {
                results[axis] = new AffineRange(domains[axis].Offset, domains[axis].Extent);
            }
        }

        var affinemap = new AffineMap(domains, default, results);
        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(input, AffineMap.Identity(rank), out var intile)
            .Write(output, affinemap, out var outTile)
            .Body(TIR.F.NTT.Unpack(intile, outTile, unpack.Lanes, unpack.Axes))
            .Build();
    }
}
