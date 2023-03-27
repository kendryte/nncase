// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.PatternMatch;
using static Nncase.IR.F.NN;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldHardSwish : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "div",
            "divCall",
            BinaryOp.Div,
            IsBinary(
                "mul",
                "mulCall",
                BinaryOp.Mul,
                IsWildcard("input"),
                IsClamp(
                    "clamp",
                    "clampCall",
                    IsBinary(
                        "add",
                        "addCall",
                        BinaryOp.Add,
                        IsWildcard(),
                        IsTensorConst("addConst") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) }),
                    IsTensorConst("clampMin") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) },
                    IsTensorConst("clampMax") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) })),
            IsTensorConst("divConst") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) });

    private Expr? GetReplace(Expr input, Call addCall, Tensor<float> divConst, Tensor<float> addConst, Tensor<float> clampMin, Tensor<float> clampMax)
    {
        if (addCall[Binary.Lhs] == input
            && Math.Abs(divConst[0] - 6f) < 1e-6f
            && Math.Abs(addConst[0] - 3f) < 1e-6f
            && Math.Abs(clampMin[0]) < 1e-6f
            && Math.Abs(clampMax[0] - 6f) < 1e-6)
        {
            return HardSwish(input);
        }

        return null;
    }
}
