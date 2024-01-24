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

using static Nncase.IR.F.CPU;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.CPU.Affine;

[RuleGenerator]
public partial class LowerUnary : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsUnary(
      target_name: "unary",
      _ => true,
      IsWildcard("input") with { TypePattern = HasFixedShape() });

    private Expr GetReplace(Unary unary, Expr input)
    {
        var rank = input.CheckedShape.Rank;
        return IR.F.Affine.Grid()
            .Read(input, AffineMap.Identity(rank), out var inTile)
            .Write(TIR.T.CreateBuffer(input.CheckedTensorType, TIR.MemoryLocation.Data, out _), AffineMap.Identity(rank), out _)
            .Body(IR.F.Math.Unary(unary.UnaryOp, inTile))
            .Build();
    }
}
