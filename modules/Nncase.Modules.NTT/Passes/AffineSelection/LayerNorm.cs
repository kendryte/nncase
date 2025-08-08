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
    public Expr SelectLayerNorm(Op op, Call call, Expr output)
    {
        if (output.CheckedShape is not { Rank: > 0 })
        {
            return call;
        }

        int axis = 0;
        if (op is IR.NN.LayerNorm layerNorm)
        {
            axis = layerNorm.Axis;
        }
        else if (op is IR.NTT.PackedLayerNorm packedLayerNorm)
        {
            axis = packedLayerNorm.Axis;
        }

        var input = (Expr)call.Arguments[0];
        var scale = (Expr)call.Arguments[1];
        var bias = (Expr)call.Arguments[2];
        var inputShape = (RankedShape)input.CheckedShape;
        var rank = input.CheckedShape.Rank;
        if (!inputShape[axis..].ToArray().All(d => d is DimConst))
        {
            return call;
        }

        var domains = IR.F.Affine.Domains(rank);
        var results = domains.Select(x => new AffineRange(x.Offset, x.Extent)).ToArray();
        var inputAccess = new AffineMap(domains, default, Enumerable.Range(0, rank).Select(i => i < axis ? new AffineRange(domains[i].Offset, domains[i].Extent) : new AffineRange(domains[i].Offset, inputShape[i].FixedValue)).ToArray());
        var scaleAccess = new AffineMap(domains, default, Enumerable.Range(axis, rank - axis).Select(i => new AffineRange(domains[i].Offset, inputShape[i].FixedValue)).ToArray());
        var biasAccess = new AffineMap(domains, default, Enumerable.Range(axis, rank - axis).Select(i => new AffineRange(domains[i].Offset, inputShape[i].FixedValue)).ToArray());

        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(input, inputAccess, out var inTile)
            .Read(scale, scaleAccess, out var scaleTile)
            .Read(bias, biasAccess, out var biasTile)
            .Write(output, inputAccess, out var outTile)
            .Body(op switch
            {
                IR.NN.LayerNorm ln => TIR.F.NTT.PackedLayerNorm(inTile, scaleTile, biasTile, outTile, ln.Axis, ln.Epsilon, ln.UseMean, Array.Empty<int>(), Array.Empty<Dimension>()),
                IR.NTT.PackedLayerNorm ln => TIR.F.NTT.PackedLayerNorm(inTile, scaleTile, biasTile, outTile, ln.Axis, ln.Epsilon, ln.UseMean, ln.PackedAxes, ((RankedShape)call[IR.NTT.PackedLayerNorm.PadedNums]).Dimensions.ToArray()),
                _ => throw new NotSupportedException($"Unsupported layer norm operation: {op}"),
            })
            .Build();
    }
}
