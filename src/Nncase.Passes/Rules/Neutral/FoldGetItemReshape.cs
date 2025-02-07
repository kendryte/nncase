// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using GetItem = Nncase.IR.Tensors.GetItem;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class FoldGetItemReshape : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsGetItem(null, "getItem", ReshapePattern, new long[] { 0 });

    public Pattern ReshapePattern => IsReshape(IsWildcard("input"), new long[] { 1 });

    private Expr? GetReplace(Expr input, int index)
    {
        return input;
    }
}
