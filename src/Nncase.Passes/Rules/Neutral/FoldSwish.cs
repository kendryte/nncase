// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public sealed partial class FoldSwishPattern1 : RewriteRule<Pattern>
{
    public FoldSwishPattern1()
    {
        var input = IsWildcard("input");
        Pattern = IsSwappableBinary(null, null, b => b.BinaryOp == BinaryOp.Mul, IsSigmoid(input), input);
    }

    /// <inheritdoc/>
    public override Pattern Pattern { get; }

    private Expr? GetReplace(Expr input)
    {
        return IR.F.NN.Swish(input);
    }
}

[RuleGenerator]
public sealed partial class FoldSwishPattern2 : RewriteRule<Pattern>
{
    public FoldSwishPattern2()
    {
        var input = IsWildcard("input");
        Pattern = IsSwappableBinary(null, null, b => b.BinaryOp == BinaryOp.Mul, IsSigmoid(IsSwappableBinary(null, null, b => b.BinaryOp == BinaryOp.Mul, input, IsTensorConst("beta", IsFloatScalar()))), input);
    }

    /// <inheritdoc/>
    public override Pattern Pattern { get; }

    private Expr? GetReplace(Expr input, TensorConst beta)
    {
        return IR.F.NN.Swish(input, beta);
    }
}
