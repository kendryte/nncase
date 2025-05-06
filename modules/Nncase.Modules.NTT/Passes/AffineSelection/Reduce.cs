// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public sealed partial class NTTAffineSelectionPass
{
    private Expr SelectReduce(IR.NTT.PackedReduce reduce, Call call, Expr output)
    {
        var input = (Expr)call[IR.NTT.PackedReduce.Input];
        if (output.CheckedShape is not { IsFixed: true, Rank: > 0 }
            || reduce.ReduceOp == ReduceOp.Mean)
        {
            return call;
        }

        var inputShape = input.CheckedShape.ToValueArray();
        var rank = inputShape.Length;
        var domains = IR.F.Affine.Domains(rank);
        var outrank = call.CheckedShape.Rank;
        var results = new AffineRange[outrank];
        {
            var j = 0;
            for (int i = 0; i < rank; i++)
            {
                if (reduce.Axes.Contains(i))
                {
                    if (reduce.KeepDims == true)
                    {
                        results[j++] = new AffineRange(0, 1);
                    }
                }
                else
                {
                    results[j++] = new AffineRange(domains[i].Offset, domains[i].Extent);
                }
            }
        }

        var affinemap = new AffineMap(domains, default, results);
        return IR.F.Affine.Grid()
            .Domain(rank, out var domainVar)
            .Read(input, AffineMap.Identity(rank), out var intile)
            .Write(output, affinemap, out var outTile)
            .Body(TIR.F.NTT.Reduce(intile, outTile, GetLoadPreviousExpr(reduce.Axes, domainVar), reduce.PackedAxes.ToArray(), reduce.PadedNums.ToArray(), reduce.Axes, reduce.KeepDims, reduce.ReduceOp))
            .Build();
    }

    private Expr GetLoadPreviousExpr(IRArray<int> axes, Expr domainVar)
    {
        Expr? outExpr = null;
        foreach (var axis in axes)
        {
            var domainAxisVar = (Expr)domainVar[axis][0];
            if (outExpr is null)
            {
                outExpr = IR.F.Math.NotEqual(domainAxisVar, 0L);
            }
            else
            {
                outExpr = IR.F.Math.LogicalAnd(outExpr, IR.F.Math.NotEqual(domainAxisVar, 0L));
            }
        }

        if (outExpr is null)
        {
            throw new NotSupportedException("reduce axes is empty");
        }

        return outExpr;
    }
}
