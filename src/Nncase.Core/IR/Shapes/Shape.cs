// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance.Helpers;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR.Tensors;
using Nncase.Utilities;

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
public sealed class Shape : Expr, IEquatable<Shape?>, IReadOnlyList<Dimension>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(ReadOnlySpan<Dimension> dimensions)
        : base(dimensions.ToArray())
    {
        RefreshKind();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// init from the dimensions
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(ReadOnlySpan<int> dimensions)
        : this(dimensions.AsValueEnumerable().Select(x => (Dimension)(long)x).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(ReadOnlySpan<long> dimensions)
        : this(dimensions.AsValueEnumerable().Select(i => (Dimension)i).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(params int[] dimensions)
        : this(dimensions.AsSpan())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(params long[] dimensions)
        : this(dimensions.AsSpan())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(params Dimension[] dimensions)
        : this(dimensions.AsSpan())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(IEnumerable<Dimension> dimensions)
        : this(dimensions.ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(IEnumerable<int> dimensions)
        : this(dimensions.Select(x => (Dimension)(long)x).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(IEnumerable<long> dimensions)
        : this(dimensions.Select(x => (Dimension)x).ToArray())
    {
    }

    private Shape(ShapeKind kind)
        : base(Array.Empty<Expr>())
    {
        Kind = kind;
    }

    /// <summary>
    /// Gets an invalid shape.
    /// </summary>
    public static Shape Invalid { get; } = new Shape(ShapeKind.Invalid);

    /// <summary>
    /// Gets an unranked shape.
    /// </summary>
    public static Shape Unranked { get; } = new Shape(ShapeKind.Unranked);

    /// <summary>
    /// Gets a scalar shape.
    /// </summary>
    public static Shape Scalar { get; } = new Shape(ShapeKind.Fixed);

    /// <summary>
    /// Gets dimensions.
    /// </summary>
    public ReadOnlySpan<Dimension> Dimensions => SpanUtility.UnsafeCast<Expr, Dimension>(Operands);

    /// <summary>
    /// Gets kind.
    /// </summary>
    public ShapeKind Kind { get; private set; }

    /// <summary>
    /// Gets a value indicating whether is readonly.
    /// </summary>
    public bool IsReadOnly => true;

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
    public bool IsScalar => IsFixed && Dimensions.Length == 0;

    /// <summary>
    /// Gets rank.
    /// </summary>
    public int Rank => IsRanked ? Dimensions.Length : throw new InvalidOperationException("Shape is unranked");

    /// <summary>
    /// Gets get Total Elements.
    /// </summary>
    public long Size => Enumerable.Range(0, Rank).Aggregate(1L, (size, i) => size * this[i].FixedValue);

    /// <inheritdoc/>
    public int Count => Operands.Length;

    /// <summary>
    /// Gets the dimension.
    /// </summary>
    /// <param name="index">Index, allowing negative value.</param>
    /// <returns>Dimension.</returns>
    public new Dimension this[int index] => index >= 0 ? Dimensions[index] : Dimensions[Rank + index];

    public new Dimension this[long index] => this[(int)index];

    public new Dimension this[Index index] => Dimensions[index];

    public ReadOnlySpan<Dimension> this[System.Range range] => Dimensions[range];

    public static implicit operator ReadOnlySpan<long>(Shape shape) => shape.Select(x => x.FixedValue).ToArray();

    public static implicit operator Shape(int[] dimensions) => new Shape(dimensions);

    public static implicit operator Shape(long[] dimensions) => new Shape(dimensions);

    public static implicit operator Shape(Dimension[] dimensions) => new Shape(dimensions);

    public static Shape operator +(Shape lhs, Shape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            throw new ArgumentException($"Shape {lhs} and {rhs} rank not match");
        }

        return new Shape(lhs.Dimensions.AsValueEnumerable().Select((l, i) => l + rhs[i]).ToArray());
    }

    public static Shape operator -(Shape lhs, Shape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            throw new ArgumentException($"Shape {lhs} and {rhs} rank not match");
        }

        return new Shape(lhs.Dimensions.AsValueEnumerable().Select((l, i) => l - rhs[i]).ToArray());
    }

    public static Shape operator *(Shape lhs, Shape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            throw new ArgumentException($"Shape {lhs} and {rhs} rank not match");
        }

        return new Shape(lhs.Dimensions.AsValueEnumerable().Select((l, i) => l * rhs[i]).ToArray());
    }

    public static Shape operator /(Shape lhs, Shape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            throw new ArgumentException($"Shape {lhs} and {rhs} rank not match");
        }

        return new Shape(lhs.Dimensions.AsValueEnumerable().Select((l, i) => l / rhs[i]).ToArray());
    }

    public static Shape operator %(Shape lhs, Shape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            throw new ArgumentException($"Shape {lhs} and {rhs} rank not match");
        }

        return new Shape(lhs.Dimensions.AsValueEnumerable().Select((l, i) => l % rhs[i]).ToArray());
    }

    public static Shape operator +(Shape lhs, Dimension rhs)
    {
        return new Shape(lhs.Dimensions.AsValueEnumerable().Select(x => x + rhs).ToArray());
    }

    public static Shape operator -(Shape lhs, Dimension rhs)
    {
        return new Shape(lhs.Dimensions.AsValueEnumerable().Select(x => x - rhs).ToArray());
    }

    public static Shape operator *(Shape lhs, Dimension rhs)
    {
        return new Shape(lhs.Dimensions.AsValueEnumerable().Select(x => x * rhs).ToArray());
    }

    public static Shape operator /(Shape lhs, Dimension rhs)
    {
        return new Shape(lhs.Dimensions.AsValueEnumerable().Select(x => x / rhs).ToArray());
    }

    public static Shape operator %(Shape lhs, Dimension rhs)
    {
        return new Shape(lhs.Dimensions.AsValueEnumerable().Select(x => x % rhs).ToArray());
    }

    public static Shape operator +(Shape lhs, int rhs) => lhs + (Dimension)rhs;

    public static Shape operator -(Shape lhs, int rhs) => lhs - (Dimension)rhs;

    public static Shape operator *(Shape lhs, int rhs) => lhs * (Dimension)rhs;

    public static Shape operator /(Shape lhs, int rhs) => lhs / (Dimension)rhs;

    public static Shape operator %(Shape lhs, int rhs) => lhs % (Dimension)rhs;

    public static Shape operator +(Shape lhs, long rhs) => lhs + (Dimension)rhs;

    public static Shape operator -(Shape lhs, long rhs) => lhs - (Dimension)rhs;

    public static Shape operator *(Shape lhs, long rhs) => lhs * (Dimension)rhs;

    public static Shape operator /(Shape lhs, long rhs) => lhs / (Dimension)rhs;

    public static Shape operator %(Shape lhs, long rhs) => lhs % (Dimension)rhs;

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
    public static Shape Unknown(int rank)
    {
        return new Shape(Enumerable.Range(0, rank).Select(x => Dimension.Unknown));
    }

    public static Shape Repeat(Dimension value, int length) => new Shape(Enumerable.Repeat(value, length).ToArray());

    public static Shape Range(Dimension start, int rank)
    {
        return new Shape(Enumerable.Range(0, rank).Select(x => start + x).ToArray());
    }

    public IEnumerator<Dimension> GetEnumerator()
    {
        for (int i = 0; i < Count; i++)
        {
            yield return Dimensions[i];
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>
    /// Get Prod.
    /// </summary>
    public Dimension Prod()
    {
        return new DimProduct(Dimensions.ToArray()).Simplify();
    }

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

    /// <summary>
    /// return new shape after insert dim.
    /// </summary>
    public Shape InsertAndClone(int index, Dimension dim)
    {
        var l = Dimensions.AsValueEnumerable().ToList();
        l.Insert(index, dim);
        return new Shape(l.ToArray());
    }

    /// <summary>
    /// return new shape after insert dim.
    /// </summary>
    public Shape InsertAndClone(int index, IEnumerable<Dimension> dims)
    {
        var l = Dimensions.AsValueEnumerable().ToList();
        foreach (var d in dims)
        {
            l.Insert(index++, d);
        }

        return new Shape(l.ToArray());
    }

    /// <summary>
    /// convert to the int list.
    /// </summary>
    public List<long> ToValueList()
    {
        return this.Select(x => x.FixedValue).ToList();
    }

    /// <summary>
    /// convert the int array.
    /// </summary>
    public long[] ToValueArray()
    {
        return this.Select(x => x.FixedValue).ToArray();
    }

    public Expr ToValueArrayExpr()
    {
        if (IsFixed)
        {
            return ToValueArray();
        }

        var tuple = new IR.Tuple(Dimensions.AsValueEnumerable().Select(x => x.ToValueExpr()).ToArray());
        return IR.F.Tensors.Stack(tuple, 0);
    }

    /// <inheritdoc/>
    public override string ToString() => Kind switch
    {
        ShapeKind.Invalid => "[invalid]",
        ShapeKind.Unranked => "[*]",
        _ => $"[{StringUtility.Join(',', Dimensions)}]",
    };

    /// <inheritdoc/>
    public bool Equals(Shape? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Dimensions.SequenceEqual(other.Dimensions);
    }

    /// <inheritdoc/>
    public override bool Equals(object? other)
    {
        return other is Shape shape && Equals(shape);
    }

    public bool IsAssignableFrom(Shape shape)
    {
        if (IsUnranked)
        {
            return true;
        }

        if (shape.IsUnranked || Rank != shape.Rank)
        {
            return false;
        }

        for (int i = 0; i < Dimensions.Length; i++)
        {
            if (!this[i].IsAssignableFrom(shape[i]))
            {
                return false;
            }
        }

        return true;
    }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitShape(this, context);

    public Shape With(Dimension[]? dimensions = null) => new Shape(dimensions ?? Dimensions);

    protected override int GetHashCodeCore()
    {
        return HashCode.Combine(GetType(), Kind, HashCode<Expr>.Combine(Operands));
    }

    protected override void OnOperandsReplaced()
    {
        base.OnOperandsReplaced();
        RefreshKind();
    }

    private void RefreshKind()
    {
        Kind = this.All(x => x.IsFixed) ? ShapeKind.Fixed : ShapeKind.HasUnknownDimension;
    }
}
