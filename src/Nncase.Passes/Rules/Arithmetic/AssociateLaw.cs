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
    public IPattern Pattern { get; } = IsWildcard("x") + IsWildcard("y") + IsWildcard("z");

    private Expr? GetReplace(Expr x, Expr y, Expr z)
    {
        return x + (y + z);
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
