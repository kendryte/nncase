// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using DryIoc.ImTools;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldNopSelect : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsSelect(
        "select",
        "call",
        IsTensorConst("condition"),
        IsWildcard("lhs"),
        IsWildcard("rhs"));

    private Expr? GetReplace(bool condition, Expr lhs, Expr rhs)
    {
        return condition ? lhs : rhs;
    }
}

[RuleGenerator]
public sealed partial class FoldCompareSelect : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsSelect(
        "select",
        "call2",
        IsCompare(
            "compare",
            "call1",
            _ => true,
            IsWildcard("clhs"),
            IsTensorConst("crhs")),
        IsTensorConst("slhs"),
        IsWildcard("srhs"));

    private Expr? GetReplace(Compare compare, Expr clhs, Tensor crhs, Tensor slhs, Expr srhs)
    {
        if (compare.CompareOp == CompareOp.Equal && crhs.ToArray<float>()[0] == 1 && slhs.ToArray<float>()[0] == 1)
        {
            if (Equals(clhs, srhs))
            {
                return clhs;
            }
        }

        return null;
    }
}
