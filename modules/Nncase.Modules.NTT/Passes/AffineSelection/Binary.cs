// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.Targets;
using Nncase.TIR;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectBinary(Binary binary, Call call, Expr output)
    {
        var lhs = (Expr)call[Binary.Lhs];
        var rhs = (Expr)call[Binary.Rhs];
        if (lhs.CheckedShape is not { Rank: > 0 } || rhs.CheckedShape is not { Rank: > 0 })
        {
            return call;
        }

        var lhsShape = lhs.CheckedShape;
        var rhsShape = rhs.CheckedShape;
        if (!TryGetBinaryAffineRelation(lhsShape, rhsShape, out var domains, out var lhsMap, out var rhsMap))
        {
            return call;
        }

        return IR.F.Affine.Grid()
            .Domain(domains.Length, out var _)
            .Read(lhs, lhsMap, out var lhsTile)
            .Read(rhs, rhsMap, out var rhsTile)
            .Write(output, AffineMap.Identity(domains.Length), out var outTile)
            .Body(TIR.F.NTT.Binary(binary.BinaryOp, lhsTile, rhsTile, outTile))
            .Build();
    }
}
