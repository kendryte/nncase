// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.F.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class PowOf2ToSquare : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
    IsBinary(
        "pow",
        "call",
        p => p.BinaryOp is BinaryOp.Pow,
        IsWildcard("input"),
        IsTensorConst("power"));

    private Expr? GetReplace(Expr input, TensorConst power)
    {
        if (power.Value.ToArray<float>().All(x => x == 2))
        {
            return Unary(UnaryOp.Square, input);
        }

        return null;
    }
}
