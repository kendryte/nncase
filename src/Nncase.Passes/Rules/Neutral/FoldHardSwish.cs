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
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldHardSwish1 : RewriteRule<CallPattern>
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

[RuleGenerator]
public sealed partial class FoldHardSwish2 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "mul",
            "mulCall",
            BinaryOp.Mul,
            IsBinary(
                "mul1_6",
                "mul1_6Call",
                BinaryOp.Mul,
                IsTensorConst("mul1_6Const") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) },
                IsRelu6(
                    "relu6",
                    "relu6Call",
                    IsBinary(
                        "add",
                        "addCall",
                        BinaryOp.Add,
                        IsWildcard(),
                        IsTensorConst("addConst") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) }))),
            IsWildcard("input"));

    private Expr? GetReplace(Expr input, Call addCall, Tensor<float> mul1_6Const, Tensor<float> addConst)
    {
        if (addCall[Binary.Lhs] == input
            && Math.Abs(addConst[0] - 3f) < 1e-6f
            && Math.Abs(mul1_6Const[0] - 0.1666666716337204f) < 1e-6f)
        {
            return HardSwish(input);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldHardSwish3 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "mul",
            "mulCall",
            BinaryOp.Mul,
            IsWildcard("input"),
            IsBinary(
                "div6",
                "div6Call",
                BinaryOp.Div,
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
                    IsTensorConst("clampMax") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) }),
                IsTensorConst("div6Const") with { TypePattern = HasRank(0) | HasShape(new[] { 1 }) }));

    private Expr? GetReplace(Expr input, Call addCall, Tensor<float> div6Const, Tensor<float> addConst, Tensor<float> clampMin, Tensor<float> clampMax)
    {
        if (addCall[Binary.Lhs] == input
            && Math.Abs(addConst[0] - 3f) < 1e-6f
            && Math.Abs(div6Const[0] - 6f) < 1e-6f
            && Math.Abs(clampMin[0]) < 1e-6f
            && Math.Abs(clampMax[0] - 6f) < 1e-6f)
        {
            return HardSwish(input);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldHardSwish4 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "mul",
            "mulCall",
            BinaryOp.Mul,
            IsWildcard(),
            IsHardSigmoid(
                "hardSigmoid",
                "hardSigmoidCall",
                IsWildcard("input")));

    private Expr? GetReplace(Expr input, Call mulCall, Call hardSigmoidCall)
    {
        if (mulCall[Binary.Lhs] == input)
        {
            return HardSwish(input);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class FoldHardSwish5 : RewriteRule<CallPattern>
{
    /// <inheritdoc/>
    public override CallPattern Pattern { get; } =
        IsBinary(
            "mul",
            "mulCall",
            BinaryOp.Mul,
            IsHardSigmoid(
                "hardSigmoid",
                "hardSigmoidCall",
                IsWildcard("input")),
            IsWildcard());

    private Expr? GetReplace(Expr input, Call mulCall, Call hardSigmoidCall)
    {
        if (mulCall[Binary.Rhs] == input)
        {
            return HardSwish(input);
        }

        return null;
    }
}
