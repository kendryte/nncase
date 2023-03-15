// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.F;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// (x * y) * z => x * (y * z).
/// </summary>
[RuleGenerator]
public sealed partial class ReassociateMul : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsWildcard("x") * IsWildcard("y") * IsWildcard("z");

    private Expr? GetReplace(Expr x, Expr y, Expr z) => x * (y * z);
}

/// <summary>
/// x * 1 => x.
/// </summary>
[RuleGenerator]
public sealed partial class Xmul1 : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsWildcard("x") * 1.0;

    private Expr? GetReplace(Expr x) => x;
}

/// <summary>
/// x / x => 1.
/// </summary>
[RuleGenerator]
public sealed partial class XDivX : IRewriteRule
{
    /// <summary>
    /// Initializes a new instance of the <see cref="XDivX"/> class.
    /// </summary>
    public XDivX()
    {
        var x = IsWildcard("x") with { TypePattern = IsTensor() };
        Pattern = x / x;
    }

    /// <inheritdoc/>
    public IPattern Pattern { get; }

    private Expr? GetReplace(Expr x)
    {
        var value = ((Tensor)1).CastTo(x.CheckedDataType);
        return Tensors.ConstantOfShape(Tensors.ShapeOf(x), value);
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
