// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class TileToExpand : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsTile(
        "tile",
        "call",
        _ => true,
        IsWildcard("input") with { TypePattern = HasFixedShape() },
        IsTensorConst("repeats"));

    private Expr? GetReplace(Call call, Expr input, long[] repeats, RunPassContext context)
    {
        var inShape = input.CheckedShape.ToValueArray();
        if (Enumerable.Range(0, repeats.Length).All(i => repeats[i] == 1 || inShape[i] == 1))
        {
            return IR.F.Tensors.Expand(input, call.CheckedShape.ToValueArray());
        }

        return null;
    }
}