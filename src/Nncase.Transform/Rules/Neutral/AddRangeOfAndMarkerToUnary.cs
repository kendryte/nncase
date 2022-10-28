
// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Numerics.Tensors;
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
using Unary = Nncase.IR.Math.Unary;
using Shape = Nncase.IR.Shape;
using F = Nncase.IR.F;

namespace Nncase.Transform.Rules.Neutral;

/// <summary>
/// Insert RangeOf and RangeOfMarker
/// </summary>
[RuleGenerator]
public sealed partial class AddRangeOfAndMarkerToUnary : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsUnary("unary", "call",
            u => u.UnaryOp is UnaryOp.Abs or UnaryOp.Neg,
            IsWildcard("input"));
    private Expr? GetReplace(Unary unary, Call call, Expr input, RunPassOptions options)
    {
        var output = F.Math.Unary(unary.UnaryOp, IR.F.Math.RangeOfMarker(input, IR.F.Math.RangeOf(input)));
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}