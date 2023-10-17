// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Ncnn;
using Nncase.PatternMatch;

using static Nncase.IR.F.Ncnn;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Ncnn;

[RuleGenerator]
public partial class LowerUnary : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsUnary(
      target_name: "unary",
      _ => true,
      IsWildcard("input") with { TypePattern = IsFloat() & HasRank(x => x <= 3) });

    private static UnaryOperationType? MapUnaryOp(UnaryOp unaryOp) =>
        unaryOp switch
        {
            UnaryOp.Abs => UnaryOperationType.ABS,
            UnaryOp.Acos => UnaryOperationType.ACOS,
            _ => null,
        };

    private Expr? GetReplace(Unary unary, Expr input)
    {
        if (MapUnaryOp(unary.UnaryOp) is UnaryOperationType op)
        {
            var newInput = new Var(input.CheckedType);
            return new Call(new Fusion("ncnn", NcnnUnary(newInput, op), new[] { newInput }), input);
        }

        return null;
    }
}
