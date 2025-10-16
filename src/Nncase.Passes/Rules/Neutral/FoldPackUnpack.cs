// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldPackUnpack : RewriteRule<Pattern>
{
    public override Pattern Pattern { get; } = PatternMatch.F.Tensors.IsPack("vectorize", "caller", _ => true, PatternMatch.F.Tensors.IsUnpack("devectorize", "callee", _ => true, IsWildcard("input")));

    private Expr? GetReplace(IR.Tensors.Pack vectorize, IR.Tensors.Unpack devectorize, Expr input)
    {
        if (vectorize.Axes.SequenceEqual(devectorize.Axes) && vectorize.Lanes.SequenceEqual(devectorize.Lanes))
        {
            return input;
        }

        return null;
    }
}
