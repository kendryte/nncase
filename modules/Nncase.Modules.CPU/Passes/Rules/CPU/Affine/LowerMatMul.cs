// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU.Affine;

[RuleGenerator]
public partial class LowerMatmul : RewriteRule<Pattern>
{
    public LowerMatmul(string moduleKind = CPUTarget.Kind)
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; }

    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsCall(
        "call",
        IsOp<Op>("op", op => op is TIR.CPU.Matmul),
        IsWildcard("lhs") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") },
        IsWildcard("rhs") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") },
        IsWildcard("output") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") },
        IsWildcard("loadC"));

    private Expr? GetReplace(Expr call, TIR.CPU.Matmul op, Expr lhs, Expr rhs, Expr output)
    {
        // TODO: summa not support tiling for now.
        if (lhs.CheckedType is DistributedType ldt && ldt.AxisPolices.Last() is SBPSplit)
        {
            return null;
        }

        var lhsShape = lhs.CheckedShape.ToValueArray();
        var rhsShape = rhs.CheckedShape.ToValueArray();
        var rank = Math.Max(lhs.CheckedShape.Rank, rhs.CheckedShape.Rank) + 1;
        var domains = IR.F.Affine.Domains(rank);
        var lhsRes = new AffineRange[lhsShape.Length];
        var rhsRes = new AffineRange[rhsShape.Length];
        for (int i = rank - 1 - 3; i >= 0; i--)
        {
            var lhsi = i - (rank - (lhsShape.Length + 1));
            var rhsi = i - (rank - (rhsShape.Length + 1));
            switch (lhsi, rhsi)
            {
                case (>= 0, >= 0):
                    switch (lhsShape[lhsi], rhsShape[rhsi])
                    {
                        case (long a, long b) when a == b:
                            lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                            rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                            break;
                        case (1, _):
                            lhsRes[lhsi] = new AffineRange(0, 1);
                            rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                            break;
                        case (_, 1):
                            lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                            rhsRes[rhsi] = new AffineRange(0, 1);
                            break;
                        default:
                            return null;
                    }

                    break;
                case (< 0, >= 0):
                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                    break;
                case (>= 0, < 0):
                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                    break;
                case (< 0, < 0):
                    break;
            }
        }

        var (om, ok, on) = (rank - 3, rank - 2, rank - 1);
        var (lm, lk) = (lhsShape.Length - 2, lhsShape.Length - 1);
        var (rk, rn) = (rhsShape.Length - 2, rhsShape.Length - 1);
        if (op.TransposeA)
        {
            (lm, lk) = (lk, lm);
        }

        if (op.TransposeB)
        {
            (rk, rn) = (rn, rk);
        }

        lhsRes[lm] = new AffineRange(domains[om].Offset, domains[om].Extent);
        lhsRes[lk] = new AffineRange(domains[ok].Offset, domains[ok].Extent);
        rhsRes[rk] = lhsRes[lk];
        rhsRes[rn] = new AffineRange(domains[on].Offset, domains[on].Extent);

        var lhsMap = new AffineMap(domains, default, lhsRes);
        var rhsMap = new AffineMap(domains, default, rhsRes);
        var outMap = new AffineMap(domains, default, domains.SkipLast(3).Concat([domains[om], domains[on]]).Select(x => new AffineRange(x.Offset, x.Extent)).ToArray());
        return IR.F.Affine.Grid(ModuleKind)
            .Domain(rank, out var domainVar)
            .Read(lhs, lhsMap, out var lhsTile)
            .Read(rhs, rhsMap, out var rhsTile)
            .Write(output, outMap, out var outTile)
            .Body(
                TIR.F.CPU.Matmul(lhsTile, rhsTile, outTile, IR.F.Math.NotEqual(domainVar[ok][0], 0L), op.LhsPackedAxes, op.LhsPadedNums, op.RhsPackedAxes, op.RhsPadedNums, op.TransposeA, op.TransposeB, op.FusedReduce))
            .Build();
    }
}
