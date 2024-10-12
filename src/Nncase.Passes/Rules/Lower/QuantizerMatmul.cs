// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Lower;

[RuleGenerator]
public sealed partial class QuantizerMatmul : IRewriteRule
{
    public IPattern Pattern { get; }
        = IsRangeOfMarker(
            "markerC",
            IsMatMul(
                "matmul",
                "call",
                _ => true,
                IsRangeOfMarker("markerA", IsWildcard("inputA"), IsConst("scaleA")),
                IsRangeOfMarker("markerB", IsWildcard("inputB"), IsConst("scaleB"))),
            IsWildcard("scaleC"));

    private Expr? GetReplace(Expr matmul, Call call, Expr markerA, Const scaleA, Expr markerB, Const scaleB, Expr markerC, RunPassContext context)
    {
        return null;
    }
}
