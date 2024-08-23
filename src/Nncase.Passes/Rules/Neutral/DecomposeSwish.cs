// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Utilities;

using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Decompose swish.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposeSwish : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsSwish(
            "swish",
            "swishCall",
            _ => true,
            IsWildcard("input"),
            IsTensorConst("beta"));

    private Expr? GetReplace(Expr input, Call swishCall, float beta)
    {
        return IR.F.Math.Div(input, 1f + IR.F.Math.Exp(-beta * input));
    }
}
