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
public partial class LowerBinary : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = PatternMatch.F.Math.IsBinary(
      "binary",
      "call",
      op => PassUtility.IsCpuSupported(op),
      IsWildcard("lhs") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") },
      IsWildcard("rhs") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") });

    private Expr? GetReplace(Expr call, Binary binary, Expr lhs, Expr rhs)
    {
        // [8, 16] * [16]
        var lhsShape = lhs.CheckedShape.ToValueArray();
        var rhsShape = rhs.CheckedShape.ToValueArray();
        var rank = Math.Max(lhs.CheckedShape.Rank, rhs.CheckedShape.Rank);
        var domains = IR.F.Affine.Domains(rank);
        var lhsRes = new AffineRange[lhs.CheckedShape.Rank];
        var rhsRes = new AffineRange[rhs.CheckedShape.Rank];
        for (int i = rank - 1; i >= 0; i--)
        {
            var lhsi = i - (rank - lhs.CheckedShape.Rank);
            var rhsi = i - (rank - rhs.CheckedShape.Rank);
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

        var lhsMap = new AffineMap(domains, default, lhsRes);
        var rhsMap = new AffineMap(domains, default, rhsRes);
        var outBuffer = call.CheckedType switch
        {
            TensorType t => IR.F.Buffer.Uninitialized(t.DType, TIR.MemoryLocation.Data, t.Shape.ToValueArray()),
            DistributedType dt => IR.F.Buffer.Uninitialized(dt.TensorType.DType, TIR.MemoryLocation.Data, dt.TensorType.Shape.ToValueArray(), dt.NdSBP, dt.Placement),
            _ => throw new ArgumentOutOfRangeException(nameof(call)),
        };
        return IR.F.Affine.Grid(CPUTarget.Kind)
            .Read(lhs, lhsMap, out var lhsTile)
            .Read(rhs, rhsMap, out var rhsTile)
            .Write(outBuffer, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.CPU.Binary(binary.BinaryOp, lhsTile, rhsTile, outTile))
            .Build();
    }
}

[RuleGenerator]
public partial class LowerPackedBinary : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = PatternMatch.F.CPU.IsPackedBinary(
      "binary",
      "call",
      op => PassUtility.IsCpuSupported(op),
      IsWildcard("lhs") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") },
      IsWildcard("rhs") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") });

    private Expr? GetReplace(Expr call, IR.CPU.PackedBinary binary, Expr lhs, Expr rhs)
    {
        // [8, 16] * [16]
        var lhsShape = lhs.CheckedShape.ToValueArray();
        var rhsShape = rhs.CheckedShape.ToValueArray();
        var rank = Math.Max(lhs.CheckedShape.Rank, rhs.CheckedShape.Rank);
        var domains = IR.F.Affine.Domains(rank);
        var lhsRes = new AffineRange[lhs.CheckedShape.Rank];
        var rhsRes = new AffineRange[rhs.CheckedShape.Rank];
        for (int i = rank - 1; i >= 0; i--)
        {
            var lhsi = i - (rank - lhs.CheckedShape.Rank);
            var rhsi = i - (rank - rhs.CheckedShape.Rank);
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

        var lhsMap = new AffineMap(domains, default, lhsRes);
        var rhsMap = new AffineMap(domains, default, rhsRes);
        var outBuffer = call.CheckedType switch
        {
            TensorType t => IR.F.Buffer.Uninitialized(t.DType, TIR.MemoryLocation.Data, t.Shape.ToValueArray()),
            DistributedType dt => IR.F.Buffer.Uninitialized(dt.TensorType.DType, TIR.MemoryLocation.Data, dt.TensorType.Shape.ToValueArray(), dt.NdSBP, dt.Placement),
            _ => throw new ArgumentOutOfRangeException(nameof(call)),
        };
        return IR.F.Affine.Grid(CPUTarget.Kind)
            .Read(lhs, lhsMap, out var lhsTile)
            .Read(rhs, rhsMap, out var rhsTile)
            .Write(outBuffer, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.CPU.PackedBinary(lhsTile, rhsTile, outTile, binary.BinaryOp, binary.LhsPackedAxes, binary.LhsPadedNums, binary.RhsPackedAxes, binary.RhsPadedNums))
            .Build();
    }
}
