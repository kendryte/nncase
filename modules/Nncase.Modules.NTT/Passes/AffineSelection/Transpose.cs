// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectTranspose(IR.Tensors.Transpose transpose, Call call, Expr output)
    {
        var input = (Expr)call[IR.Tensors.Transpose.Input];
        var perm = (Shape)call[IR.Tensors.Transpose.Perm];
        if (output.CheckedShape is not { IsFixed: true, Rank: > 0 }
            || !perm.IsFixed)
        {
            return call;
        }

        var permValues = perm.ToValueArray();
        var rank = input.CheckedShape.Rank;
        var domains = IR.F.Affine.Domains(rank);
        var results = new AffineRange[rank];
        for (int i = 0; i < rank; i++)
        {
            results[permValues[i]] = new AffineRange(domains[i].Offset, domains[i].Extent);
        }

        var inputAccessMap = new AffineMap(domains, default, results);
        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(input, inputAccessMap, out var intile)
            .Write(output, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.NTT.Transpose(intile, outTile, permValues.ToInts()))
            .Build();
    }
}
