// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Constant of shape.
/// </summary>
public sealed class ShapeConst : Const, IEquatable<ShapeConst?>
{
    public ShapeConst(Shape shape)
        : base(new TensorType(DataTypes.Int64, new[] { shape.Rank }))
    {
        Value = shape;
    }

    public Shape Value { get; }

    /// <inheritdoc/>
    public override string ToString()
    {
        return Value.ToString();
    }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitShapeConst(this, context);

    public ShapeConst With(Shape? value = null)
    {
        return new ShapeConst(value ?? Value);
    }

    public bool Equals(ShapeConst? other) => other is ShapeConst o && Value.Equals(o.Value);

    public override bool Equals(object? obj)
    {
        return Equals(obj as ShapeConst);
    }
}

/// <summary>
/// Constant of tensor.
/// </summary>
public sealed class DimensionConst : Const, IEquatable<DimensionConst?>
{
    public DimensionConst(Dimension value)
        : base(new TensorType(DataTypes.Int64, Shape.Scalar))
    {
        Value = value;
    }

    public Dimension Value { get; }

    /// <inheritdoc/>
    public override string ToString()
    {
        return Value.ToString();
    }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitDimensionConst(this, context);

    public DimensionConst With(Dimension? value = null)
    {
        return new DimensionConst(value ?? Value);
    }

    public bool Equals(DimensionConst? other) => other is DimensionConst o && Value.Equals(o.Value);

    public override bool Equals(object? obj)
    {
        return Equals(obj as DimensionConst);
    }
}
