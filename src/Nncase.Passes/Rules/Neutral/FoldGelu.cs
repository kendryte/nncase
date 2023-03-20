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
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldGeluWithScale : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "mul1",
            "Mul1Call",
            BinaryOp.Mul,
            IsBinary(
                "mul2",
                "mul2Call",
                BinaryOp.Mul,
                IsBinary(
                    "mul3",
                    "mul3Call",
                    BinaryOp.Mul,
                    IsWildcard("input"),
                    IsTensorConst("mul3Const") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) }),
                IsBinary(
                    "add",
                    "addCall",
                    BinaryOp.Add,
                    IsErf(
                        "erf",
                        "erfCall",
                        IsBinary(
                            "div",
                            "divCall",
                            BinaryOp.Div,
                            IsWildcard(),
                            IsTensorConst("divConst") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) })),
                    IsTensorConst("addConst") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) })),
            IsTensorConst("mul1Const") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) });

    private Expr? GetReplace(Expr input, Call mul3Call, Tensor<float> mul3Const, Call divCall, Tensor<float> divConst, Tensor<float> addConst, Tensor<float> mul1Const)
    {
        if (divCall[Binary.Lhs] == mul3Call
        && Math.Abs(mul3Const[0] - 0.5773502588272095) < 1e-6
        && Math.Abs(mul1Const[0] - 0.5f) < 1e-6f
        && Math.Abs(addConst[0] - 1f) < 1e-6f
        && Math.Abs(divConst[0] - 1.4142135381698608f) < 1e-6f)
        {
            return Gelu(input, mul3Const);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldGeneralGelu : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "mul1",
            "Mul1Call",
            BinaryOp.Mul,
            IsBinary(
                "mul2",
                "mul2Call",
                BinaryOp.Mul,
                IsWildcard("input"),
                IsBinary(
                    "add",
                    "addCall",
                    BinaryOp.Add,
                    IsErf(
                        "erf",
                        "erfCall",
                        IsBinary(
                            "div",
                            "divCall",
                            BinaryOp.Div,
                            IsWildcard(),
                            IsTensorConst("divConst") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) })),
                    IsTensorConst("addConst") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) })),
            IsTensorConst("mul1Const") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) });

    private Expr? GetReplace(Expr input, Call divCall, Tensor<float> divConst, Tensor<float> addConst, Tensor<float> mul1Const)
    {
        if (divCall[Binary.Lhs] == input && Math.Abs(mul1Const[0] - 0.5f) < 1e-6f && Math.Abs(addConst[0] - 1f) < 1e-6f && Math.Abs(divConst[0] - 1.4142135381698608f) < 1e-6f)
        {
            return Gelu(input, 1f);
        }

        return null;
    }
}
