// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.Passes.Rules.Lower.BroadcastMarkerHelper;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.Lower;

// e.g. matmul(reshape(marker(x))) -> matmul(marker(reshape(marker(x))))
[RuleGenerator]
public partial class BroadcastInputMarker : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsCall("outer", IsWildcard("outerTarget"), IsVArgsRepeat("outerParams", exprs =>
    {
        var patterns = new Pattern[exprs.Length];
        for (int i = 0; i < exprs.Length; i++)
        {
            patterns[i] = GetInputPattern(i);
        }

        return patterns;
    }));

    public Pattern GetInputPattern(int i) =>
     IsAlt(
        IsCallWildcard(
            $"input_{i}",
            IsOp<Op>($"input_target_{i}", NotChangeRangeOp),
            IsRangeOfMarker($"input_marker_{i}", IsWildcard($"marker_target_{i}"), IsWildcard($"marker_attribute_{i}"))),
        IsWildcard($"input_{i}"));

    public Expr? GetReplace(Call outer, Expr outerTarget, IReadOnlyList<Expr> outerParams, IMatchResult result)
    {
        if (!Enumerable.Range(0, outerParams.Count).Select(i => result.GetValueOrDefault($"input_marker_{i}")).Any(e => e is not null))
        {
            return null;
        }

        var newArgs = new Expr[outerParams.Count];
        for (int i = 0; i < outerParams.Count; i++)
        {
            if (result.GetValueOrDefault($"input_marker_{i}") is Marker marker && result[$"marker_target_{i}"] is Expr target && result[$"marker_attribute_{i}"] is Expr range)
            {
                newArgs[i] = IR.F.Math.RangeOfMarker(outerParams[i], range).With(mixQuantInfo: marker.MixQuantInfo, adaQuantInfo: marker.AdaQuantInfo);
            }
            else
            {
                newArgs[i] = outerParams[i];
            }
        }

        return new Call(outerTarget, newArgs);
    }
}

// e.g. marker(reshape(matmul(x))) -> marker(reshape(marker(matmul(x))))
[RuleGenerator]
public partial class BroadcastOutputMarker : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsRangeOfMarker(
        "marker",
        IsCallWildcard("output", IsOp<Op>("outputTarget", NotChangeRangeOp), IsCallWildcard("input", IsWildcard("inputTarget"))),
        IsWildcard("range"));

    public Expr? GetReplace(Marker marker, Expr range, Call output, Op outputTarget, IReadOnlyList<Expr> outputParams)
    {
        return ReplaceCallFirstParam(outputTarget, outputParams, IR.F.Math.RangeOfMarker(outputParams[0], range).With(adaQuantInfo: marker.AdaQuantInfo, mixQuantInfo: marker.MixQuantInfo));
    }
}

internal static class BroadcastMarkerHelper
{
    public static bool NotChangeRangeOp(Op op)
    {
        return op is Squeeze || op is Unsqueeze || op is Reshape || op is Broadcast;
    }
}
