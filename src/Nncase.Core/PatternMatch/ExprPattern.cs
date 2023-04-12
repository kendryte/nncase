// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Expr"/>.
/// </summary>
/// <param name="Condition">Expression condition.</param>
/// <param name="Name">name.</param>
public sealed record ExprPattern(Func<Expr, bool> Condition, string? Name) : Pattern<Expr>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ExprPattern"/> class.
    /// </summary>
    /// <param name="name">name.</param>
    public ExprPattern(string? name)
        : this(x => true, name)
    {
        IsWildcard = true;
    }

    /// <summary>
    /// Gets a value indicating whether is wildcard pattern.
    /// </summary>
    public bool IsWildcard { get; }

    /// <inheritdoc/>
    protected override bool MatchLeafCore(Expr expr) => Condition(expr);
}

public static partial class Utility
{
    /// <summary>
    /// Get the current expr checked Shape.
    /// </summary>
    /// <param name="expr">expr.</param>
    /// <returns>dimension.</returns>
    /// <exception cref="InvalidOperationException">e.</exception>
    public static List<Dimension> GetShape(Expr expr) => expr.CheckedType switch
    {
        TensorType type => new List<Dimension>(type.Shape),
        _ => throw new InvalidOperationException($"The Expr {expr.GetType().Name} Has No Shape!"),
    };

    /// <summary>
    /// build wildcard pattern.
    /// </summary>
    public static ExprPattern IsWildcard(string? name) => new ExprPattern(name);

    /// <summary>
    /// fast utitlty for build condition.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="condition">conditions.</param>
    public static ExprPattern IsWildcard(string? name, Func<Expr, bool> condition) => new ExprPattern(condition, name);

    /// <summary>
    /// <see cref="IsWildcard(string?)"/>.
    /// </summary>
    public static ExprPattern IsWildcard() => IsWildcard(null);

    /// <summary>
    /// fast utitlty for build optional none pattern.
    /// </summary>
    public static ExprPattern IsNone() => new ExprPattern(e => e is None, null);
}
