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
/// (x * y) * z => x * (y * z).
/// </summary>
[RuleGenerator]
public sealed partial class AssociateMul : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsWildcard("x") * IsWildcard("y") * IsWildcard("z");

    private Expr? GetReplace(Expr x, Expr y, Expr z) => x * (y * z);
}

/// <summary>
/// (x + y) + z = x + (y + z).
/// </summary>
[RuleGenerator]
public sealed partial class AssociateAdd : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsBinary("b1", "b1Call", op => op.BinaryOp is BinaryOp.Add or BinaryOp.Sub, IsBinary("b0", "b0Call", op => op.BinaryOp is BinaryOp.Add or BinaryOp.Sub, IsWildcard("x"), IsWildcard("y")), IsWildcard("z"));

    private Expr? GetReplace(Binary b0, Binary b1, Expr x, Expr y, Expr z)
    {
        Expr nx = x;
        Expr ny = b0.BinaryOp == BinaryOp.Add ? y : -y;
        Expr nz = b1.BinaryOp == BinaryOp.Add ? z : -z;
        return nx + (ny + nz);
    }
}

/// <summary>
/// (x * y) / z => x * (y / z).
/// </summary>
[RuleGenerator]
public sealed partial class ReassociateDiv : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsWildcard("x") * IsWildcard("y") / IsWildcard("z") with { TypePattern = IsFloat() };

    private Expr? GetReplace(Expr x, Expr y, Expr z) => x * (y / z);
}
