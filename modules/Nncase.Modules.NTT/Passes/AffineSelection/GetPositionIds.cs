// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.NN;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public static Expr SelectGetPositionIds(GetPositionIds op, Call call, Expr output)
    {
        var kvcache = (Expr)call[GetPositionIds.KVCache];
        if (output.CheckedShape is not { Rank: > 0 })
        {
            return call;
        }

        var kvcacheShape = kvcache.CheckedShape;
        var outputShape = output.CheckedShape;
        var rank = outputShape.Rank;

        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(kvcache, AffineMap.FromCallable((dims, syms) => [], rank, 0), out var inTile)
            .Write(output, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.NTT.GetPositionIds(inTile, outTile, (DistributedType)output.CheckedType))
            .Build();
    }
}
