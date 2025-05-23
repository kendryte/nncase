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
public partial class FoldGetItemConcat : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsGetItem(null, "getItem", ConcatPattern, IsDimension("index") | IsShape("index", Shape.Unknown(1)));

    public Pattern ConcatPattern => IsConcat(0, IsTuple("input"));

    private BaseExpr? GetReplace(IR.Tuple input, int index)
    {
        return input.Fields[index];
    }
}
