// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Imaging;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Imaging;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Add range of marker base class.
/// </summary>
/// <typeparam name="T"></typeparam>
[RuleGenerator]
public partial class AddRangeOfAndMarkerSingleInput : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } =
      IsCallWildcard("call",
        IsOp<Op>("op", op => op switch
        {
            (Transpose or SpaceToBatch or ActivationOp or
            ResizeImage or ReduceWindow2D or Reduce or
            Pad or BatchToSpace or Broadcast or Clamp or
            LSTM) => true,
            _ => false,
        }),
        IsWildcard("input"));

    Expr? GetReplace(Call call, Op op, Expr input, IReadOnlyList<Expr> callParams, RunPassContext context)
    {
        var newCall = ReplaceCallParams(op, callParams, (input, IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input))));
        context.MatchOptions.SuppressPattern(newCall, Pattern); // only invoke once
        return IR.F.Math.RangeOfMarker(newCall, IR.F.Math.RangeOf(newCall));
    }
}

[RuleGenerator]
public partial class AddRangeOfAndMarkerDoubleInput : RewriteRule<Pattern>
{
    /// <inheritdoc/>
    public override Pattern Pattern { get; } =
      IsCallWildcard("call",
        IsOp<Op>("op", op => op switch
        {
            (MatMul or Conv2D or Conv2DTranspose or
            Compare or Binary) => true,
            _ => false,
        }),
        IsWildcard("lhs"),
        IsWildcard("rhs"));

    Expr? GetReplace(Call call, Op op, Expr lhs, Expr rhs, IReadOnlyList<Expr> callParams, RunPassContext context)
    {
        var newCall = ReplaceCallParams(op, callParams,
          (lhs, IR.F.Math.RangeOfMarker(lhs, IR.F.Math.RangeOf(lhs))),
          (rhs, IR.F.Math.RangeOfMarker(rhs, IR.F.Math.RangeOf(rhs))));
        context.MatchOptions.SuppressPattern(newCall, Pattern); // only invoke once
        return IR.F.Math.RangeOfMarker(newCall, IR.F.Math.RangeOf(newCall));
    }
}
