// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldNopIf : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsWildcard("expr", expr => expr is If @if && @if.Condition is TensorConst);

    private Expr? GetReplace(If expr)
    {
        var cond = ((TensorConst)expr.Condition).Value.ToScalar<bool>();
        return cond ? expr.Then : expr.Else;
    }
}
