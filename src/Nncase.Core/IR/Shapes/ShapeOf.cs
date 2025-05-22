// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using CommunityToolkit.HighPerformance.Helpers;

namespace Nncase.IR.Shapes;

/// <summary>
/// Shape of expression.
/// </summary>
public sealed class ShapeOf : Shape, IEquatable<ShapeOf?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ShapeOf"/> class.
    /// </summary>
    /// <param name="value">Value.</param>
    public ShapeOf(Expr value)
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
    public override ShapeKind Kind => Value.CheckedShape.Kind;

    /// <summary>
    /// Gets rank.
    /// </summary>
    public override int Rank => Value.CheckedShape.Rank;

    public override Dimension this[Dimension index] => new DimAt(this, index);

    public static bool operator ==(ShapeOf? lhs, ShapeOf? rhs)
    {
        return EqualityComparer<ShapeOf>.Default.Equals(lhs, rhs);
    }

    public static bool operator !=(ShapeOf? lhs, ShapeOf? rhs)
    {
        return !(lhs == rhs);
    }

    public override Expr ToValueArrayExpr() => IR.F.Tensors.ShapeOf(Value);

    /// <inheritdoc/>
    public override string ToString() => $"shapeof({Value})";

    /// <inheritdoc/>
    public bool Equals(ShapeOf? other) => other is not null && Value.Equals(other.Value);

    /// <inheritdoc/>
    public override bool Equals(object? other) => Equals(other as ShapeOf);

    public override bool IsAssignableFrom(Shape shape) => true;

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitShapeOf(this, context);

    public ShapeOf With(Expr? value = null) => new ShapeOf(value ?? Value);
}
