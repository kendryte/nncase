// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldNopWhere : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } = IsWhere(
        "where",
        "call",
        false,
        IsTensorConst("condition"),
        IsWildcard("lhs"),
        IsWildcard("rhs"));

    private Expr? GetReplace(bool condition, Expr lhs, Expr rhs)
    {
        return condition ? lhs : rhs;
    }
}
