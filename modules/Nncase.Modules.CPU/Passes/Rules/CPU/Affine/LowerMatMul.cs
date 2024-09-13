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
        IsOp<Op>("op", op => op is MatMul or IR.CPU.PackedMatMul),
        IsWildcard("lhs") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") },
        IsWildcard("rhs") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") });

    private Expr? GetReplace(Expr call, Op op, Expr lhs, Expr rhs)
    {
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
#pragma warning disable SA1008 // Opening parenthesis should be spaced correctly
                case ( >= 0, >= 0):
                    switch (lhsShape[lhsi], rhsShape[rhsi])
                    {
                        case (int a, int b) when a == b:
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
                case ( < 0, >= 0):
                    rhsRes[rhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                    break;
                case ( >= 0, < 0):
                    lhsRes[lhsi] = new AffineRange(domains[i].Offset, domains[i].Extent);
                    break;
                case ( < 0, < 0):
                    break;
#pragma warning restore SA1008 // Opening parenthesis should be spaced correctly
            }
        }

        var (om, ok, on) = (rank - 3, rank - 2, rank - 1);
        var (lm, lk) = (lhsShape.Length - 2, lhsShape.Length - 1);
        var (rk, rn) = (rhsShape.Length - 2, rhsShape.Length - 1);
        if (op is IR.CPU.PackedMatMul pm)
        {
            if (pm.TransposeA)
            {
                (lm, lk) = (lk, lm);
            }

            if (pm.TransposeB)
            {
                (rk, rn) = (rn, rk);
            }
        }

        lhsRes[lm] = new AffineRange(domains[om].Offset, domains[om].Extent);
        lhsRes[lk] = new AffineRange(domains[ok].Offset, domains[ok].Extent);
        rhsRes[rk] = lhsRes[lk];
        rhsRes[rn] = new AffineRange(domains[on].Offset, domains[on].Extent);

        var lhsMap = new AffineMap(domains, default, lhsRes);
        var rhsMap = new AffineMap(domains, default, rhsRes);
        var outBuffer = call.CheckedType switch
        {
            TensorType t => IR.F.Buffer.Uninitialized(t.DType, TIR.MemoryLocation.Data, t.Shape.ToValueArray()),
            DistributedType dt => IR.F.Buffer.Uninitialized(dt.TensorType.DType, TIR.MemoryLocation.Data, dt.TensorType.Shape.ToValueArray(), dt.NdSBP, dt.Placement),
            _ => throw new ArgumentOutOfRangeException(nameof(call)),
        };
        return IR.F.Affine.Grid(ModuleKind)
            .Domain(rank, out var domainVar)
            .Read(lhs, lhsMap, out var lhsTile)
            .Read(rhs, rhsMap, out var rhsTile)
            .Write(outBuffer, new AffineMap(domains, default, domains.SkipLast(2).Concat(domains.TakeLast(1)).Select(x => new AffineRange(x.Offset, x.Extent)).ToArray()), out var outTile)
            .Body(op switch
            {
                MatMul => TIR.F.CPU.Matmul(lhsTile, rhsTile, outTile, IR.F.Math.NotEqual(domainVar[rank - 2][0], 0L)),
                IR.CPU.PackedMatMul pop => TIR.F.CPU.Matmul(lhsTile, rhsTile, outTile, IR.F.Math.NotEqual(domainVar[rank - 2][0], 0L), pop.LhsPackedAxes, pop.LhsPadedNums, pop.RhsPackedAxes, pop.RhsPadedNums, pop.TransposeA, pop.TransposeB),
                _ => throw new System.Diagnostics.UnreachableException(),
            })
            .Build();
    }
}
