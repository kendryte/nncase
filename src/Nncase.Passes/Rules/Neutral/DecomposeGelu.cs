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
/// Decompose Gelu.
/// </summary>
[RuleGenerator]
public sealed partial class DecomposeGelu : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } =
        IsGelu(
            "gelu",
            "geluCall",
            _ => true,
            IsWildcard("input"),
            IsTensorConst("alpha"));

    private Expr? GetReplace(Expr input, Call geluCall, float alpha)
    {
        var scaledInput = IR.F.Math.Mul(input, alpha);
        return IR.F.Math.Mul(new[] { 0.5f }, IR.F.Math.Mul(scaledInput, IR.F.Math.Add(IR.F.NN.Erf(IR.F.Math.Div(scaledInput, IR.F.Math.Sqrt(new[] { 2f }))), 1f)));
    }
}
