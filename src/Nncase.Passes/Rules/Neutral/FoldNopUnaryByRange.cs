// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class FoldNopAbsByRange : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsUnary(
        "unaryOp",
        "unary",
        UnaryOp.Abs,
        IsWildcard("input"));

    private Expr? GetReplace(Expr input)
    {
        if (input.Metadata.Range?.Min >= 0)
        {
            return input;
        }

        return null;
    }
}
