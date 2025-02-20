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
        IsTensorConst("rhs"));

    private Expr? GetReplace(Compare compareOp, Expr lhs, Tensor rhs)
    {
        if (compareOp.CompareOp == CompareOp.Equal && rhs.ElementType == DataTypes.Int64 && rhs.Cast<long>().All(x => x == -1))
        {
            return false;
        }

        return null;
    }
}
