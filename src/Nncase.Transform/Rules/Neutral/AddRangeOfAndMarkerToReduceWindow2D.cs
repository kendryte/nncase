
// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using Tensorflow;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;
using Shape = Nncase.IR.Shape;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Insert RangeOf and RangeOfMarker
/// </summary>
[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToRedeceWindow2D : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsReduceWindow2D("reduceWindow2D", "call", _ => true,
            IsWildcard("input"),
            IsWildcard("initvalue"),
            IsWildcard("filter"),
            IsWildcard("stride"),
            IsWildcard("padding"),
            IsWildcard("dilation"),
            IsWildcard("ceilmode"),
            IsWildcard("countincludepad"));

    private Expr? GetReplace(ReduceWindow2D reduceWindow2D, Expr input, Expr initvalue, Expr filter, Expr stride, Expr padding,
        Expr dilation, Expr ceilmode, Expr countincludepad, RunPassOptions options)
    {
        var output = ReduceWindow2D(reduceWindow2D.ReduceOp, IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)), initvalue, filter, stride, padding, dilation, ceilmode, countincludepad);
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}