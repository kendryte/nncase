// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.TIR;
using Pack = Nncase.IR.Tensors.Pack;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    private Expr SelectPad(Pad pad, Call call, Expr output)
    {
        var input = (Expr)call[Pad.Input];
        if (output.CheckedShape is not { Rank: > 0 })
        {
            return call;
        }

        var inputShape = input.CheckedShape;
        var rank = inputShape.Rank;
        var domains = IR.F.Affine.Domains(rank);
        var results = new AffineRange[rank];

        var paddings = (Paddings)call[IR.NN.Pad.Pads];
        var actualPadAxes = Enumerable.Range(0, paddings.Count).Where(i => !(paddings[i] is { IsFixed: true } pad && pad.Sum() == 0)).ToArray();
        for (int axis = 0; axis < rank; axis++)
        {
            results[axis] = new AffineRange(domains[axis].Offset, domains[axis].Extent);
        }

        var affinemap = new AffineMap(domains, default, results);
        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(input, affinemap, out var intile)
            .Write(output, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.NTT.Pad(intile, outTile, (Paddings)call[Pad.Pads], ((TensorConst)call[Pad.Value]).Value.ToScalar<float>(), actualPadAxes))
            .Build();
    }
}
