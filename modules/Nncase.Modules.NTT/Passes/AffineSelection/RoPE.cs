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

public partial class NTTAffineSelectionPass
{
    public Expr SelectRoPE(IR.NTT.VectorizedRoPE rope, Call call, Expr output)
    {
        var input = (Expr)call[IR.NTT.VectorizedRoPE.Input];
        var cos = (Expr)call[IR.NTT.VectorizedRoPE.Cos];
        var sin = (Expr)call[IR.NTT.VectorizedRoPE.Sin];

        var rank = input.CheckedShape.Rank;
        var domains = IR.F.Affine.Domains(rank);
        var inOutResults = domains.Select(x => new AffineRange(x.Offset, x.Extent)).ToArray();
        var inOutMap = new AffineMap(domains, default, inOutResults);

        // [head, dim, seq]
        var sinCosResults = inOutResults[1..];
        var sinCosMap = new AffineMap(domains, default, sinCosResults);

        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(input, inOutMap, out var inTile)
            .Read(sin, sinCosMap, out var sinTile)
            .Read(cos, sinCosMap, out var cosTile)
            .Write(output, inOutMap, out var outTile)
            .Body(TIR.F.NTT.RoPE(inTile, cosTile, sinTile, outTile))
            .Build();
    }
}
