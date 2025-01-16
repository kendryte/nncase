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
public partial class FoldNopCompareByRange : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsCompare(
        "compareOp",
        "compare",
        CompareOp.Equal,
        IsWildcard("lhs"),
        IsWildcard("rhs"));

    private Expr? GetReplace(Expr lhs, Expr rhs)
    {
        if (lhs.Metadata.Range?.Max < rhs.Metadata.Range?.Min
            || lhs.Metadata.Range?.Min > rhs.Metadata.Range?.Max)
        {
            return false;
        }

        return null;
    }
}
