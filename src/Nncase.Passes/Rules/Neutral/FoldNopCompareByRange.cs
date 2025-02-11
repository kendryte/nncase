// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
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
        _ => true,
        IsWildcard("lhs"),
        IsWildcard("rhs"));

    private Expr? GetReplace(Compare compareOp, Expr lhs, Expr rhs)
    {
        var lhsRange = lhs.Metadata.Range ?? ValueRange<double>.Full;
        var rhsRange = rhs.Metadata.Range ?? ValueRange<double>.Full;

        if (lhsRange.Max < rhsRange.Min)
        {
            return compareOp.CompareOp switch
            {
                CompareOp.NotEqual => true,
                CompareOp.LowerThan => true,
                CompareOp.LowerOrEqual => true,
                CompareOp.Equal => false,
                CompareOp.GreaterOrEqual => false,
                CompareOp.GreaterThan => false,
                _ => null,
            };
        }
        else if (lhsRange.Min > rhsRange.Max)
        {
            return compareOp.CompareOp switch
            {
                CompareOp.NotEqual => true,
                CompareOp.LowerThan => false,
                CompareOp.LowerOrEqual => false,
                CompareOp.Equal => false,
                CompareOp.GreaterOrEqual => true,
                CompareOp.GreaterThan => true,
                _ => null,
            };
        }

        return null;
    }
}
