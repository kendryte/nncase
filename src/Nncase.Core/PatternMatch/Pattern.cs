// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern.
/// </summary>
/// <param name="Name">this pattern's indenitifer name.</param>
public abstract partial record Pattern(string? Name) : IPattern
{
    /// <summary>
    /// Gets or sets hashcode cache, for speedup get hashcode.
    /// </summary>
    protected int? HashCode { get; set; }

    public virtual bool Equals(Pattern? other)
    {
        return !(other is null) && EqualityContract == other.EqualityContract;
    }

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode ??=
      System.HashCode.Combine(EqualityComparer<Type>.Default.GetHashCode(EqualityContract));

    /// <inheritdoc/>
    public abstract bool MatchLeaf(Expr input);

    /// <summary>
    /// Print members.
    /// </summary>
    /// <param name="builder">String builder.</param>
    /// <returns>Break print.</returns>
    protected virtual bool PrintMembers(StringBuilder builder)
    {
        builder.Append(this.DumpAsIL());
        return true;
    }
}

/// <summary>
///
/// Pattern.
/// </summary>
/// <typeparam name="TExpr">Expression type.</typeparam>
public record Pattern<TExpr>(string? Name) : Pattern(Name), IPattern<TExpr>
    where TExpr : Expr
{
    /// <summary>
    /// Gets pattern for CheckedType, defulat match IR Type.
    /// </summary>
    public TypePattern? TypePattern { get; init; }

    /// <inheritdoc/>
    public bool MatchLeaf(TExpr expr)
    {
        return MatchCheckedType(expr) && MatchLeafCore(expr);
    }

    /// <inheritdoc/>
    public sealed override bool MatchLeaf(Expr input) => input is TExpr expr && MatchLeaf(expr);

    /// <summary>
    /// Match leaf impl.
    /// </summary>
    /// <param name="expr">Input expression.</param>
    /// <returns>Match result.</returns>
    protected virtual bool MatchLeafCore(TExpr expr) => true;

    /// <summary>
    /// Match The Expr Type.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Is Matched.</returns>
    private bool MatchCheckedType(Expr expr) => (TypePattern, expr.CheckedType) switch
    {
        (null, _) => true,
        (TypePattern pattern, IRType type) => pattern.MatchLeaf(type),
        _ => false,
    };
}
