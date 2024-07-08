// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Arithmetic;

/// <summary>
/// x * y => y * x.
/// </summary>
[RuleGenerator]
public sealed partial class CommutateMul : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsWildcard("x") * IsWildcard("y");

    private Expr? GetReplace(Expr x, Expr y) => y * x;
}

/// <summary>
/// x * y => y * x.
/// </summary>
[RuleGenerator]
public sealed partial class CommutateAdd : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsBinary("b0", "b0Call", op => op.BinaryOp is BinaryOp.Add or BinaryOp.Sub, IsWildcard("x"), IsWildcard("y"));

    private Expr? GetReplace(Binary b0, Expr x, Expr y)
    {
        return b0.BinaryOp == BinaryOp.Add ? y + x : (-y) + x;
    }
}
