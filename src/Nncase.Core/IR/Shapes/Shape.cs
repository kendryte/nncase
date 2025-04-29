// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase.IR.Shapes;

namespace Nncase.IR;

/// <summary>
/// Shape kind.
/// </summary>
public enum ShapeKind
{
    /// <summary>
    /// Invalid shape.
    /// </summary>
    Invalid,

    /// <summary>
    /// Unranked shape.
    /// </summary>
    Unranked,

    /// <summary>
    /// Shape contains unknown dimensions.
    /// </summary>
    HasUnknownDimension,

    /// <summary>
    /// Fixed shape.
    /// </summary>
    Fixed,
}

/// <summary>
/// Tensor shape.
/// </summary>
[CollectionBuilder(typeof(ShapeBuilder), nameof(ShapeBuilder.Create))]
public abstract class Shape : BaseExpr, IEnumerable<Dimension>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    public Shape(BaseExpr[] operands)
        : base(operands)
    {
    }

    /// <summary>
    /// Gets an invalid shape.
    /// </summary>
    public static InvalidShape Invalid { get; } = new InvalidShape();

    /// <summary>
    /// Gets an unranked shape.
    /// </summary>
    public static UnrankedShape Unranked { get; } = new UnrankedShape(None.Default);

    /// <summary>
    /// Gets a scalar shape.
    /// </summary>
    public static RankedShape Scalar { get; } = new RankedShape(Array.Empty<Dimension>());

    /// <summary>
    /// Gets kind.
    /// </summary>
    public abstract ShapeKind Kind { get; }

    /// <summary>
    /// Gets a value indicating whether fixed.
    /// </summary>
    public bool IsFixed => Kind == ShapeKind.Fixed;

    /// <summary>
    /// Gets a value indicating whether invalid.
    /// </summary>
    public bool IsInvalid => Kind == ShapeKind.Invalid;

    /// <summary>
    /// Gets a value indicating whether unranked.
    /// </summary>
    public bool IsUnranked => Kind == ShapeKind.Unranked;

    /// <summary>
    /// Gets a value indicating whether has unknown dimension.
    /// </summary>
    public bool HasUnknownDimension => Kind == ShapeKind.HasUnknownDimension;

    /// <summary>
    /// Gets a value indicating whether ranked.
    /// </summary>
    public bool IsRanked => IsFixed || HasUnknownDimension;

    /// <summary>
    /// Gets a value indicating whether scalar.
    /// </summary>
    public bool IsScalar => IsFixed && Rank == 0;

    /// <summary>
    /// Gets rank.
    /// </summary>
    public abstract int Rank { get; }

    /// <summary>
    /// Gets the dimension.
    /// </summary>
    /// <param name="index">Index, allowing negative value.</param>
    /// <returns>Dimension.</returns>
    public virtual Dimension this[int index] => throw new NotSupportedException("Shape is not indexed");

    public Dimension this[long index] => this[(int)index];

    public virtual Dimension this[Index index] => throw new NotSupportedException("Shape is not indexed");

    public override Dimension this[Dimension index] => throw new NotSupportedException("Shape is not indexed");

    public static implicit operator Shape(int[] dimensions) => (RankedShape)dimensions;

    public static implicit operator Shape(long[] dimensions) => (RankedShape)dimensions;

    public static implicit operator Shape(Dimension[] dimensions) => new RankedShape(dimensions);

    public static implicit operator Shape(Tensor<long> tensor) => (RankedShape)tensor;

    public static implicit operator Shape(Tensor<int> tensor) => (RankedShape)tensor;

    public static bool operator ==(Shape? lhs, Shape? rhs)
    {
        return EqualityComparer<Shape>.Default.Equals(lhs, rhs);
    }

    public static bool operator !=(Shape? lhs, Shape? rhs)
    {
        return !(lhs == rhs);
    }

    /// <summary>
    /// Gets a shape with rank unknwon dimension.
    /// </summary>
    public static RankedShape Unknown(int rank)
    {
        return new RankedShape(Enumerable.Range(0, rank).Select(x => Dimension.Unknown));
    }

    public static RankedShape Repeat(Dimension value, int length) => new RankedShape(Enumerable.Repeat(value, length).ToArray());

    public static RankedShape Range(Dimension start, int rank)
    {
        return new RankedShape(Enumerable.Range(0, rank).Select(x => start + x).ToArray());
    }

    /// <inheritdoc/>
    public override bool Equals(object? other) => Equals(other as Shape);

    public abstract bool IsAssignableFrom(Shape shape);

    public long ProdWithDynamicAsMaxValue(int dynamicValue = short.MaxValue, long scale = 1)
    {
        if (CompilerServices.TryGetMaxShape(this, out var maxShape))
        {
            return Enumerable.Range(0, Rank).Aggregate(scale, (acc, x) => acc * maxShape[x]);
        }
        else
        {
            return dynamicValue;
        }
    }

    public virtual long[] ToValueArray() => throw new InvalidOperationException("Shape is not fixed");

    public abstract Expr ToValueArrayExpr();

    public virtual IEnumerator<Dimension> GetEnumerator() => throw new NotSupportedException("Shape is not indexed");

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
