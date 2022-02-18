// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Pattern;

/// <summary>
/// Pattern.
/// </summary>
public abstract partial record Pattern : IPattern
{
    /// <summary>
    /// Gets or sets hashcode cache, for speedup get hashcode.
    /// </summary>
    protected int? HashCode { get; set; }

    /// <inheritdoc/>
    public override int GetHashCode() => HashCode ??=
      System.HashCode.Combine(EqualityComparer<Type>.Default.GetHashCode(EqualityContract));

    /// <inheritdoc/>
    public abstract bool Match(object input);

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
/// Pattern.
/// </summary>
/// <typeparam name="TExpr">Expression type.</typeparam>
public record Pattern<TExpr>(Func<TExpr, bool> Condition) : Pattern, IPattern<TExpr>
    where TExpr : Expr
{
    /// <summary>
    /// Gets pattern for CheckedType, defulat match IR Type.
    /// </summary>
    public TypePattern TypePattern { get; init; } = TypePatternUtility.IsIRType();

    /// <inheritdoc/>
    public bool Match(TExpr expr)
    {
        return MatchCheckedType(expr) && Condition(expr);
    }

    /// <inheritdoc/>
    public sealed override bool Match(object input) => input is TExpr expr && Match(expr);

    /// <summary>
    /// Match The Expr Type.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Is Matched.</returns>
    private bool MatchCheckedType(Expr expr) => expr.CheckedType switch
    {
        IRType type => TypePattern.MatchLeaf(type),
        _ => false,
    };
}
