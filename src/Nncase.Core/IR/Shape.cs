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

public record struct FixedAndDynamicDimension(long Fixed, Dimension? Dynamic)
{
    public static implicit operator FixedAndDynamicDimension((long Fixed, Dimension? Dynamic) value) => new FixedAndDynamicDimension(value.Fixed, value.Dynamic);

    public static FixedAndDynamicDimension operator *(FixedAndDynamicDimension a, FixedAndDynamicDimension b)
    {
        var dyn = (a.Dynamic, b.Dynamic) switch
        {
            (null, null) => (Dimension?)null,
            (null, Dimension x) => x,
            (Dimension x, null) => x,
            (Dimension x, Dimension y) => x * y,
        };
        return new FixedAndDynamicDimension(a.Fixed * b.Fixed, dyn);
    }

    public static FixedAndDynamicDimension operator /(FixedAndDynamicDimension a, long b)
    {
        if (a.Fixed % b == 0 || a.Dynamic is null)
        {
            return new FixedAndDynamicDimension(a.Fixed / b, a.Dynamic);
        }

        return new FixedAndDynamicDimension(1, a.Fixed * a.Dynamic.Value / b);
    }

    public static FixedAndDynamicDimension operator /(FixedAndDynamicDimension a, FixedAndDynamicDimension b)
    {
        if (a.Fixed % b.Fixed == 0)
        {
            return (a.Dynamic, b.Dynamic) switch
            {
                (null, null) => new FixedAndDynamicDimension(a.Fixed / b.Fixed, null),
                (null, Dimension y) => new FixedAndDynamicDimension(1, a.Fixed / b.Fixed / y),
                (Dimension x, null) => new FixedAndDynamicDimension(a.Fixed / b.Fixed, x),
                (Dimension x, Dimension y) when x == y => new FixedAndDynamicDimension(a.Fixed / b.Fixed, null),
                (Dimension x, Dimension y) => new FixedAndDynamicDimension(1, a.Fixed / b.Fixed * x / y),
            };
        }

        return (a.Dynamic, b.Dynamic) switch
        {
            (null, null) => new FixedAndDynamicDimension(a.Fixed / b.Fixed, null),
            (null, Dimension y) => new FixedAndDynamicDimension(1, a.Fixed / (b.Fixed * y)),
            (Dimension x, null) => new FixedAndDynamicDimension(1, a.Fixed * x / b.Fixed),
            (Dimension x, Dimension y) when x == y => new FixedAndDynamicDimension(a.Fixed / b.Fixed, null),
            (Dimension x, Dimension y) => new FixedAndDynamicDimension(1, a.Fixed * x / (b.Fixed * y)),
        };
    }

    public static FixedAndDynamicDimension Abs(FixedAndDynamicDimension value) =>
        new(System.Math.Abs(value.Fixed), value.Dynamic is null ? (Dimension?)null : Dimension.Abs(value.Dynamic.Value));

    public static FixedAndDynamicDimension? TryDivExactly(FixedAndDynamicDimension a, FixedAndDynamicDimension b)
    {
        if (a.Fixed % b.Fixed == 0)
        {
            return (a.Dynamic, b.Dynamic) switch
            {
                (null, null) => new FixedAndDynamicDimension(a.Fixed / b.Fixed, null),
                (null, Dimension y) => new FixedAndDynamicDimension(1, a.Fixed / b.Fixed / y),
                (Dimension x, null) => new FixedAndDynamicDimension(a.Fixed / b.Fixed, x),
                (Dimension x, Dimension y) when x == y => new FixedAndDynamicDimension(a.Fixed / b.Fixed, null),
                (Dimension x, Dimension y) => new FixedAndDynamicDimension(1, a.Fixed / b.Fixed * x / y),
            };
        }

        return (a.Dynamic, b.Dynamic) switch
        {
            (null, _) => null,
            (Dimension x, null) => new FixedAndDynamicDimension(1, a.Fixed * x / b.Fixed),
            (Dimension x, Dimension y) when x == y => null,
            (Dimension x, Dimension y) => new FixedAndDynamicDimension(1, a.Fixed * x / (b.Fixed * y)),
        };
    }

    public Dimension ToDimension()
    {
        return (Fixed, Dynamic) switch
        {
            (_, null) => Fixed,
            (1, Dimension x) => x,
            _ => Fixed * Dynamic.Value,
        };
    }

    public Expr ToExpr()
    {
        return Dynamic is null ? Fixed : Fixed * Dynamic.Value.ToExpr();
    }
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
    public Shape(ReadOnlySpan<Expr> dimensions)
        : this(dimensions.ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// init from the dimensions
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(ReadOnlySpan<int> dimensions)
        : this(dimensions.AsValueEnumerable().Select(x => (Expr)(long)x).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(ReadOnlySpan<long> dimensions)
        : this(dimensions.AsValueEnumerable().Select(i => (Expr)i).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(ReadOnlySpan<Dimension> dimensions)
        : this(dimensions.AsValueEnumerable().Select(x => x.ToExpr()).ToArray())
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
    public Shape(IEnumerable<Expr> dimensions)
        : this(dimensions.ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(IEnumerable<int> dimensions)
        : this(dimensions.Select(x => (Expr)(long)x).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(IEnumerable<long> dimensions)
        : this(dimensions.Select(x => (Expr)x).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(IEnumerable<Dimension> dimensions)
        : this(dimensions.Select(x => x.ToExpr()).ToArray())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Shape"/> class.
    /// </summary>
    /// <param name="dimensions">Dimensions.</param>
    public Shape(params Expr[] dimensions)
        : base(dimensions.Select(CompilerServices.FastSimplifyForDimension).ToArray())
    {
        foreach (var dim in Dimensions)
        {
            var dtype = dim is Const c ? c.ValueType : dim.CheckedType;
            if (dtype != TensorType.Scalar(DataTypes.Int64)
                && dtype != NoneType.Default)
            {
                if (DumpScope.Current.IsEnabled(DumpFlags.Compile))
                {
                    DumpScope.Current.DumpIR(dim, "InvalidDimension");
                }
            }
        }

        RefreshKind();
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
    public ReadOnlySpan<Expr> Dimensions => Operands;

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

    public static implicit operator ReadOnlySpan<long>(Shape shape) => shape.Select(x => x.FixedValue).ToArray();

    public static implicit operator Shape(int[] dimensions) => new Shape(dimensions);

    public static implicit operator Shape(long[] dimensions) => new Shape(dimensions);

    public static implicit operator Shape(Dimension[] dimensions) => new Shape(dimensions);

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

    /// <summary>
    /// Gets a shape with rank unknwon dimension.
    /// </summary>
    public static Shape FromExpr(Expr value)
    {
        if (value is TensorConst tc)
        {
            return new Shape(tc.Value.ToArray<long>());
        }
        else if (value is Call { Target: Concat } concat)
        {
            if (concat.Arguments[Concat.Input.Index] is Tuple tuple)
            {
                return new Shape(tuple.Fields.AsValueEnumerable().Select(x => x[0]).ToArray());
            }
        }

        var shape = value.CheckedShape;
        if (shape.Rank != 1 || !shape.IsFixed)
        {
            // throw new ArgumentException($"Invalid shape expr: {value}", nameof(value));
            return Shape.Unranked;
        }

        var rank = (int)shape[0].FixedValue;
        return new Shape(Enumerable.Range(0, rank).Select(x => value[x]));
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
        return Enumerable.Range(0, Rank).Aggregate((Dimension)1L, (size, i) => size * this[i]);
    }

    public FixedAndDynamicDimension ProdFixedAndDynamic()
    {
        var fixedValue = 1L;
        Dimension? dynamicValue = null;
        foreach (var dim in this)
        {
            if (dim.IsFixed)
            {
                fixedValue *= dim.FixedValue;
            }
            else
            {
                dynamicValue = dynamicValue is null ? dim : dynamicValue * dim;
            }
        }

        return new(fixedValue, dynamicValue);
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
        l.Insert(index, dim.ToExpr());
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
            l.Insert(index++, d.ToExpr());
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

        var tuple = new IR.Tuple(Dimensions);
        return IR.F.Tensors.Stack(tuple, 0);
    }

    /// <inheritdoc/>
    public override string ToString() => Kind switch
    {
        ShapeKind.Invalid => "Invalid",
        ShapeKind.Unranked => "Unranked",
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

    public Shape With(Expr[]? dimensions = null) => new Shape(dimensions ?? Dimensions);

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
