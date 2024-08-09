// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.IR.F.CPU;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU.Affine;

[RuleGenerator]
public partial class LowerUnary : RewriteRule<Pattern>
{
    public LowerUnary(string moduleKind = CPUTarget.Kind)
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; }

    /// <inheritdoc/>
    public override Pattern Pattern { get; } = PatternMatch.F.Math.IsUnary(
      "unary",
      "call",
      _ => true,
      IsWildcard("input") with { TypePattern = HasShape(s => s.Rank > 0 && s.IsFixed, "tileable") });

    private Expr GetReplace(Unary unary, Expr input)
    {
        var outBuffer = input.CheckedType switch
        {
            TensorType t => IR.F.Buffer.Uninitialized(t.DType, TIR.MemoryLocation.Data, t.Shape.ToValueArray()),
            DistributedType dt => IR.F.Buffer.Uninitialized(dt.TensorType.DType, TIR.MemoryLocation.Data, dt.TensorType.Shape.ToValueArray(), dt.NdSBP, dt.Placement),
            _ => throw new ArgumentOutOfRangeException(nameof(input)),
        };

        var rank = input.CheckedShape.Rank;
        return IR.F.Affine.Grid(ModuleKind)
            .Read(input, AffineMap.Identity(rank), out var inTile)
            .Write(outBuffer, AffineMap.Identity(rank), out var outTile)
            .Body(TIR.F.CPU.Unary(unary.UnaryOp, inTile, outTile))
            .Build();
    }
}
