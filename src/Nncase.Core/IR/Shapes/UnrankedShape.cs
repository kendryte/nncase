// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using CommunityToolkit.HighPerformance.Helpers;

namespace Nncase.IR;

/// <summary>
/// Unranked shape.
/// </summary>
public sealed class UnrankedShape : Shape, IEquatable<UnrankedShape?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="UnrankedShape"/> class.
    /// </summary>
    /// <param name="value">Value.</param>
    public UnrankedShape(Expr value)
        : base([value])
    {
    }

    /// <summary>
    /// Gets dimensions.
    /// </summary>
    public Expr Value => (Expr)Operands[0];

    /// <summary>
    /// Gets kind.
    /// </summary>
    public override ShapeKind Kind => ShapeKind.Unranked;

    /// <summary>
    /// Gets rank.
    /// </summary>
    public override int Rank => throw new InvalidOperationException("Shape is unranked");

    public override Dimension this[Dimension index] => new DimAt(this, index);

    public static bool operator ==(UnrankedShape? lhs, UnrankedShape? rhs)
    {
        return EqualityComparer<UnrankedShape>.Default.Equals(lhs, rhs);
    }

    public static bool operator !=(UnrankedShape? lhs, UnrankedShape? rhs)
    {
        return !(lhs == rhs);
    }

    public override Expr ToValueArrayExpr() => Value;

    /// <inheritdoc/>
    public override string ToString() => "[*]";

    /// <inheritdoc/>
    public bool Equals(UnrankedShape? other) => other is not null && Value.Equals(other.Value);

    /// <inheritdoc/>
    public override bool Equals(object? other) => Equals(other as UnrankedShape);

    public override bool IsAssignableFrom(Shape shape) => true;

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitUnrankedShape(this, context);

    public UnrankedShape With(Expr? value = null) => new UnrankedShape(value ?? Value);
}
