
// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
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
public sealed partial class AddRangeOfAndMarkerToMatMul : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsMatMul("matmul", "call", _ => true,
            IsWildcard("lhs"),
            IsWildcard("rhs"));
    private Expr? GetReplace(MatMul matmul, Call call, Expr lhs, Expr rhs, RunPassOptions options)
    {
        var output = MatMul(IR.F.Math.RangeOfMarker(lhs, IR.F.Math.RangeOf(lhs)), IR.F.Math.RangeOfMarker(rhs, IR.F.Math.RangeOf(rhs)));
        options.MatchOptions.SuppressPattern(output, Pattern); //only invoke once
        return IR.F.Math.RangeOfMarker(output, IR.F.Math.RangeOf(output));
    }
}