// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public sealed partial class NTTAffineSelectionPass
{
    private Expr SelectCast(IR.Tensors.Cast cast, Call call, Expr output)
    {
        var input = (Expr)call[IR.Tensors.Cast.Input];
        if (output.CheckedShape is not { IsFixed: true, Rank: > 0 })
        {
            return call;
        }

        var rank = input.CheckedShape.Rank;
        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(input, AffineMap.Identity(rank), out var inTile)
            .Write(output, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.NTT.Cast(inTile, outTile, cast.NewType, cast.CastMode))
            .Build();
    }
}
