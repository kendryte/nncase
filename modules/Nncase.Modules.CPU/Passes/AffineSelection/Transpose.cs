// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.CPU;

namespace Nncase.Passes;

public partial class CPUAffineSelectionPass
{
    public Expr SelectTranspose(IR.Tensors.Transpose transpose, Call call, Expr output)
    {
        var input = call[IR.Tensors.Transpose.Input];
        var perm = call[IR.Tensors.Transpose.Perm];
        if (output.CheckedShape is not { IsFixed: true, Rank: > 0 }
            || perm is not TensorConst permConst)
        {
            return call;
        }

        var permValues = permConst.Value.Cast<int>();
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
            .Body(TIR.F.CPU.Transpose(intile, outTile, permValues.ToArray()))
            .Build();
    }
}
