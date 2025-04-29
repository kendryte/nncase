// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using CommunityToolkit.HighPerformance.Helpers;

namespace Nncase.IR;

/// <summary>
/// Invalid shape.
/// </summary>
public sealed class InvalidShape : Shape, IEquatable<InvalidShape?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="InvalidShape"/> class.
    /// </summary>
    public InvalidShape()
        : base([])
    {
    }

    /// <summary>
    /// Gets kind.
    /// </summary>
    public override ShapeKind Kind => ShapeKind.Invalid;

    /// <summary>
    /// Gets rank.
    /// </summary>
    public override int Rank => throw new InvalidOperationException("Shape is invalid");

    public override Dimension this[Dimension index] => throw new InvalidOperationException("Shape is invalid");

    public static bool operator ==(InvalidShape? lhs, InvalidShape? rhs) => true;

    public static bool operator !=(InvalidShape? lhs, InvalidShape? rhs) => false;

    public override Expr ToValueArrayExpr() => throw new InvalidOperationException("Shape is invalid");

    /// <inheritdoc/>
    public override string ToString() => "[invalid]";

    /// <inheritdoc/>
    public bool Equals(InvalidShape? other) => other is not null;

    /// <inheritdoc/>
    public override bool Equals(object? other) => Equals(other as UnrankedShape);

    public override bool IsAssignableFrom(Shape shape) => false;

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitInvalidShape(this, context);

    public InvalidShape With() => Invalid;
}
