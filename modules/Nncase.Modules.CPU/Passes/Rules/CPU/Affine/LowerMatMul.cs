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
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = PatternMatch.F.Math.IsMatMul(
      "matmul",
      "call",
      _ => true,
      IsWildcard("lhs") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") },
      IsWildcard("rhs") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") });

    private Expr? GetReplace(Expr call, MatMul matmul, Expr lhs, Expr rhs)
    {
        var lhsShape = lhs.CheckedShape.ToValueArray();
        var rhsShape = rhs.CheckedShape.ToValueArray();
        var rank = Math.Max(lhs.CheckedShape.Rank, rhs.CheckedShape.Rank) + 1;
        var domains = IR.F.Affine.Domains(rank);
        var lhsRes = new AffineRange[lhs.CheckedShape.Rank];
        var rhsRes = new AffineRange[rhs.CheckedShape.Rank];
        for (int i = rank - 1 - 3; i >= 0; i--)
        {
            var lhsi = i - (rank - (lhs.CheckedShape.Rank + 1));
            var rhsi = i - (rank - (rhs.CheckedShape.Rank + 1));
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

        lhsRes[^2] = new AffineRange(domains[^3].Offset, domains[^3].Extent);
        lhsRes[^1] = new AffineRange(domains[^2].Offset, domains[^2].Extent);
        rhsRes[^2] = lhsRes[^1];
        rhsRes[^1] = new AffineRange(domains[^1].Offset, domains[^1].Extent);

        var lhsMap = new AffineMap(domains, default, lhsRes);
        var rhsMap = new AffineMap(domains, default, rhsRes);
        var outBuffer = call.CheckedType switch
        {
            TensorType t => IR.F.Buffer.Uninitialized(t.DType, TIR.MemoryLocation.Data, t.Shape.ToValueArray()),
            DistributedType dt => IR.F.Buffer.Uninitialized(dt.TensorType.DType, TIR.MemoryLocation.Data, dt.TensorType.Shape.ToValueArray(), dt.NdSBP, dt.Placement),
            _ => throw new ArgumentOutOfRangeException(nameof(call)),
        };
        return IR.F.Affine.Grid(CPUTarget.Kind)
            .Domain(rank, out var domainVar)
            .Read(lhs, lhsMap, out var lhsTile)
            .Read(rhs, rhsMap, out var rhsTile)
            .Write(outBuffer, new AffineMap(domains, default, domains.SkipLast(2).Concat(domains.TakeLast(1)).Select(x => new AffineRange(x.Offset, x.Extent)).ToArray()), out var outTile)
            .Body(TIR.F.CPU.Matmul(lhsTile, rhsTile, outTile, IR.F.Math.Equal(domainVar[rank - 2][0], 0L)))
            .Build();
    }
}
