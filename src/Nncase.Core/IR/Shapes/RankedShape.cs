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
/// Tensor shape.
/// </summary>
public sealed class RankedShape : Shape, IEquatable<RankedShape?>, IReadOnlyList<Dimension>
{
    private ShapeKind _kind;

    /// <summary>
    /// Initializes a new instance of the <see cref="RankedShape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public RankedShape(ReadOnlySpan<Dimension> dimensions)
        : base(dimensions.AsValueEnumerable().Select(x => x as BaseExpr).ToArray())
    {
        RefreshKind();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RankedShape"/> class.
    /// init from the dimensions
    /// Initializes a new instance of the <see cref="RankedShape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public RankedShape(ReadOnlySpan<int> dimensions)
        : this(dimensions.AsValueEnumerable().Select(x => (Dimension)(long)x).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RankedShape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public RankedShape(ReadOnlySpan<long> dimensions)
        : this(dimensions.AsValueEnumerable().Select(i => (Dimension)i).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RankedShape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public RankedShape(params int[] dimensions)
        : this(dimensions.AsSpan())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RankedShape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public RankedShape(params long[] dimensions)
        : this(dimensions.AsSpan())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RankedShape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public RankedShape(params Dimension[] dimensions)
        : this(dimensions.AsSpan())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RankedShape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public RankedShape(IEnumerable<Dimension> dimensions)
        : this(dimensions.ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RankedShape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public RankedShape(IEnumerable<int> dimensions)
        : this(dimensions.Select(x => (Dimension)(long)x).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="RankedShape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public RankedShape(IEnumerable<long> dimensions)
        : this(dimensions.Select(x => (Dimension)x).ToArray())
    {
    }

    /// <summary>
    /// Gets dimensions.
    /// </summary>
    public ReadOnlySpan<Dimension> Dimensions => SpanUtility.UnsafeCast<BaseExpr, Dimension>(Operands);

    /// <summary>
    /// Gets kind.
    /// </summary>
    public override ShapeKind Kind => _kind;

    /// <summary>
    /// Gets rank.
    /// </summary>
    public override int Rank => IsRanked ? Dimensions.Length : throw new InvalidOperationException("Shape is unranked");

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
    public override Dimension this[int index] => index >= 0 ? Dimensions[index] : Dimensions[Rank + index];

    public override Dimension this[Index index] => Dimensions[index];

    public ReadOnlySpan<Dimension> this[System.Range range] => Dimensions[range];

    public override Dimension this[Dimension index]
    {
        get
        {
            index = Dimension.Positive(index, Rank);
            return index switch
            {
                DimConst dc => this[dc.Value],
                _ => new DimAt(this, index),
            };
        }
    }

    public static implicit operator ReadOnlySpan<long>(RankedShape shape) => shape.Select(x => x.FixedValue).ToArray();

    public static implicit operator RankedShape(int[] dimensions) => new RankedShape(dimensions);

    public static implicit operator RankedShape(long[] dimensions) => new RankedShape(dimensions);

    public static implicit operator RankedShape(Dimension[] dimensions) => new RankedShape(dimensions);

    public static implicit operator RankedShape(Tensor<long> tensor)
    {
        if (tensor.Length == 0)
        {
            return Scalar;
        }
        else if (tensor.Rank == 1)
        {
            return new RankedShape(tensor.ToArray());
        }
        else
        {
            throw new ArgumentException($"Shape tensor rank {tensor.Rank} not match");
        }
    }

    public static implicit operator RankedShape(Tensor<int> tensor) => tensor.Cast<long>(CastMode.KDefault);

    public static RankedShape operator +(RankedShape lhs, RankedShape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            throw new ArgumentException($"Shape {lhs} and {rhs} rank not match");
        }

        return new RankedShape(lhs.Dimensions.AsValueEnumerable().Select((l, i) => l + rhs[i]).ToArray());
    }

    public static RankedShape operator -(RankedShape lhs, RankedShape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            throw new ArgumentException($"Shape {lhs} and {rhs} rank not match");
        }

        return new RankedShape(lhs.Dimensions.AsValueEnumerable().Select((l, i) => l - rhs[i]).ToArray());
    }

    public static RankedShape operator *(RankedShape lhs, RankedShape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            throw new ArgumentException($"Shape {lhs} and {rhs} rank not match");
        }

        return new RankedShape(lhs.Dimensions.AsValueEnumerable().Select((l, i) => l * rhs[i]).ToArray());
    }

    public static RankedShape operator /(RankedShape lhs, RankedShape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            throw new ArgumentException($"Shape {lhs} and {rhs} rank not match");
        }

        return new RankedShape(lhs.Dimensions.AsValueEnumerable().Select((l, i) => l / rhs[i]).ToArray());
    }

    public static RankedShape operator %(RankedShape lhs, RankedShape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            throw new ArgumentException($"Shape {lhs} and {rhs} rank not match");
        }

        return new RankedShape(lhs.Dimensions.AsValueEnumerable().Select((l, i) => l % rhs[i]).ToArray());
    }

    public static RankedShape operator +(RankedShape lhs, Dimension rhs)
    {
        return new RankedShape(lhs.Dimensions.AsValueEnumerable().Select(x => x + rhs).ToArray());
    }

    public static RankedShape operator -(RankedShape lhs, Dimension rhs)
    {
        return new RankedShape(lhs.Dimensions.AsValueEnumerable().Select(x => x - rhs).ToArray());
    }

    public static RankedShape operator *(RankedShape lhs, Dimension rhs)
    {
        return new RankedShape(lhs.Dimensions.AsValueEnumerable().Select(x => x * rhs).ToArray());
    }

    public static RankedShape operator /(RankedShape lhs, Dimension rhs)
    {
        return new RankedShape(lhs.Dimensions.AsValueEnumerable().Select(x => x / rhs).ToArray());
    }

    public static RankedShape operator %(RankedShape lhs, Dimension rhs)
    {
        return new RankedShape(lhs.Dimensions.AsValueEnumerable().Select(x => x % rhs).ToArray());
    }

    public static RankedShape operator +(RankedShape lhs, int rhs) => lhs + (Dimension)rhs;

    public static RankedShape operator -(RankedShape lhs, int rhs) => lhs - (Dimension)rhs;

    public static RankedShape operator *(RankedShape lhs, int rhs) => lhs * (Dimension)rhs;

    public static RankedShape operator /(RankedShape lhs, int rhs) => lhs / (Dimension)rhs;

    public static RankedShape operator %(RankedShape lhs, int rhs) => lhs % (Dimension)rhs;

    public static RankedShape operator +(RankedShape lhs, long rhs) => lhs + (Dimension)rhs;

    public static RankedShape operator -(RankedShape lhs, long rhs) => lhs - (Dimension)rhs;

    public static RankedShape operator *(RankedShape lhs, long rhs) => lhs * (Dimension)rhs;

    public static RankedShape operator /(RankedShape lhs, long rhs) => lhs / (Dimension)rhs;

    public static RankedShape operator %(RankedShape lhs, long rhs) => lhs % (Dimension)rhs;

    public static bool operator ==(RankedShape? lhs, RankedShape? rhs)
    {
        return EqualityComparer<RankedShape>.Default.Equals(lhs, rhs);
    }

    public static bool operator !=(RankedShape? lhs, RankedShape? rhs)
    {
        return !(lhs == rhs);
    }

    public override IEnumerator<Dimension> GetEnumerator()
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

    /// <summary>
    /// return new shape after insert dim.
    /// </summary>
    public RankedShape InsertAndClone(int index, Dimension dim)
    {
        var l = Dimensions.AsValueEnumerable().ToList();
        l.Insert(index, dim);
        return new RankedShape(l.ToArray());
    }

    /// <summary>
    /// return new shape after insert dim.
    /// </summary>
    public RankedShape InsertAndClone(int index, IEnumerable<Dimension> dims)
    {
        var l = Dimensions.AsValueEnumerable().ToList();
        foreach (var d in dims)
        {
            l.Insert(index++, d);
        }

        return new RankedShape(l.ToArray());
    }

    /// <summary>
    /// convert the int array.
    /// </summary>
    public override long[] ToValueArray()
    {
        return this.Select(x => x.FixedValue).ToArray();
    }

    public override Expr ToValueArrayExpr()
    {
        if (IsFixed)
        {
            return ToValueArray();
        }

        var tuple = new IR.Tuple(Dimensions.AsValueEnumerable().Select(x => IR.F.Shapes.AsTensor(x)).ToArray());
        return IR.F.Tensors.Stack(tuple, 0);
    }

    /// <inheritdoc/>
    public override string ToString() => $"[{StringUtility.Join(',', Dimensions)}]";

    /// <inheritdoc/>
    public bool Equals(RankedShape? other)
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
        return other is RankedShape shape && Equals(shape);
    }

    public override bool IsAssignableFrom(Shape shape)
    {
        if (shape is RankedShape rankedShape && rankedShape.Rank == Rank)
        {
            for (int i = 0; i < Dimensions.Length; i++)
            {
                if (!this[i].IsAssignableFrom(rankedShape[i]))
                {
                    return false;
                }
            }

            return true;
        }

        return false;
    }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitRankedShape(this, context);

    public RankedShape With(Dimension[]? dimensions = null) => new RankedShape(dimensions ?? Dimensions);

    protected override int GetHashCodeCore()
    {
        return HashCode.Combine(GetType(), Kind, HashCode<BaseExpr>.Combine(Operands));
    }

    protected override void OnOperandsReplaced()
    {
        base.OnOperandsReplaced();
        RefreshKind();
    }

    private void RefreshKind()
    {
        _kind = this.All(x => x.IsFixed) ? ShapeKind.Fixed : ShapeKind.HasUnknownDimension;
    }
}
