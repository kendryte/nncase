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
public partial class LowerPack : RewriteRule<Pattern>
{
    public LowerPack(string moduleKind = CPUTarget.Kind)
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; }

    public override Pattern Pattern { get; } = IsCall(
            "call",
            IsOp<IR.CPU.Pack>("op"),
            IsWildcard("input") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") });

    private Expr? GetReplace(Expr call, IR.CPU.Pack op, Expr input)
    {
        var inputShape = input.CheckedShape.ToValueArray();
        var rank = inputShape.Length;
        var domains = IR.F.Affine.Domains(rank);
        var results = new AffineRange[rank];

        for (int axis = 0; axis < rank; axis++)
        {
            // e.g. f32[128,256] -> f32<4>[32,256]
            if (op.Axes.IndexOf(axis) is int i && i != -1)
            {
                results[axis] = new AffineRange(op.Lanes[i] * domains[axis].Offset, op.Lanes[i] * domains[axis].Extent);
            }
            else
            {
                results[axis] = new AffineRange(domains[axis].Offset, domains[axis].Extent);
            }
        }

        var affinemap = new AffineMap(domains, default, results);
        var outBuffer = call.CheckedType switch
        {
            TensorType t => IR.F.Buffer.Uninitialized(t.DType, TIR.MemoryLocation.Data, t.Shape.ToValueArray()),
            DistributedType dt => IR.F.Buffer.Uninitialized(dt.TensorType.DType, TIR.MemoryLocation.Data, dt.TensorType.Shape.ToValueArray(), dt.NdSBP, dt.Placement),
            _ => throw new ArgumentOutOfRangeException(nameof(call)),
        };

        return IR.F.Affine.Grid(ModuleKind)
            .Domain(rank, out var _)
            .Read(input, affinemap, out var intile)
            .Write(outBuffer, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.CPU.Pack(intile, outTile, op.Lanes, op.Axes))
            .Build();
    }
}
