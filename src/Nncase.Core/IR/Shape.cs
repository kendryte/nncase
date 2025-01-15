// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR.Tensors;

namespace Nncase.IR
{
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
                (null, null) => null,
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
            new(System.Math.Abs(value.Fixed), value.Dynamic is null ? null : Dimension.Abs(value.Dynamic.Value));

        public Dimension ToDimension()
        {
            return Dynamic is null ? new Dimension(Fixed) : new Dimension(Fixed) * Dynamic.Value;
        }

        public Expr ToExpr()
        {
            return Dynamic is null ? Fixed : Fixed * Dynamic.Value;
        }
    }

    /// <summary>
    /// Tensor shape.
    /// </summary>
    public sealed class Shape : IStructuralEquatable, IReadOnlyList<Dimension>, IEquatable<Shape>, IEnumerable<Dimension>
    {
        private readonly ImmutableArray<Dimension> _dimensions;

        private readonly int _hashcode;

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(ReadOnlySpan<Dimension> dimensions)
        {
            Kind = KindOf(dimensions);
            _dimensions = ImmutableArray.Create(dimensions.ToArray());
            _hashcode = StructuralComparisons.StructuralEqualityComparer.GetHashCode(_dimensions);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// init from the dimensions
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(ReadOnlySpan<int> dimensions)
            : this(dimensions.AsValueEnumerable().Select(x => new Dimension(x)).ToArray())
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(ReadOnlySpan<long> dimensions)
            : this(dimensions.AsValueEnumerable().Select(i => (int)i).ToArray())
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
        public Shape(params int[] dimensions)
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
            : this(dimensions.ToArray())
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(IEnumerable<Expr> dimensions)
            : this(dimensions.Select(i => (Dimension)i).ToArray())
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(ReadOnlySpan<Expr> dimensions)
            : this(dimensions.AsValueEnumerable().Select(i => (Dimension)i).ToArray())
        {
        }

        private Shape(ShapeKind kind, IEnumerable<Dimension> dimensions)
        {
            Kind = kind;
            _dimensions = dimensions.ToImmutableArray();
            _hashcode = StructuralComparisons.StructuralEqualityComparer.GetHashCode(_dimensions);
        }

        /// <summary>
        /// Gets an invalid shape.
        /// </summary>
        public static Shape Invalid { get; } = new Shape(ShapeKind.Invalid, new List<Dimension>());

        /// <summary>
        /// Gets an unranked shape.
        /// </summary>
        public static Shape Unranked { get; } = new Shape(ShapeKind.Unranked, new List<Dimension>());

        /// <summary>
        /// Gets a scalar shape.
        /// </summary>
        public static Shape Scalar { get; } = new Shape(ShapeKind.Fixed, new List<Dimension>());

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
        public bool IsScalar => IsFixed && _dimensions.Length == 0;

        /// <summary>
        /// Gets rank.
        /// </summary>
        public int Rank => _dimensions.Length;

        /// <summary>
        /// Gets get Total Elements.
        /// </summary>
        public long Size => Enumerable.Range(0, Rank).Aggregate(1L, (size, i) => size * _dimensions[i].FixedValue);

        /// <inheritdoc/>
        public int Count => ((IReadOnlyCollection<Dimension>)_dimensions).Count;

        /// <inheritdoc/>
        public Dimension this[int index] =>
            index >= 0
                ? ((IReadOnlyList<Dimension>)_dimensions)[index]
                : ((IReadOnlyList<Dimension>)_dimensions)[Rank + index];

        public static implicit operator ReadOnlySpan<long>(Shape shape) => shape._dimensions.Select(x => x.FixedValue).ToArray();

        public static implicit operator Shape(Dimension[] dimensions) => new Shape(dimensions);

        public static implicit operator Shape(int[] dimensions) => new Shape(dimensions);

        public static implicit operator Shape(long[] dimensions) => new Shape(dimensions);

        public static bool operator ==(Shape lhs, Shape rhs)
        {
            return lhs.Equals(rhs);
        }

        public static bool operator !=(Shape lhs, Shape rhs)
        {
            return !(lhs == rhs);
        }

        /// <summary>
        /// Gets a shape with rank unknwon dimension.
        /// </summary>
        public static Shape Unknown(int rank)
        {
            return new Shape(ShapeKind.HasUnknownDimension, Enumerable.Range(0, rank).Select(x => Dimension.Unknown()));
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
                    return new Shape(tuple.Fields);
                }
            }

            var shape = value.CheckedShape;
            if (shape.Rank != 1 || !shape.IsFixed)
            {
                // throw new ArgumentException($"Invalid shape expr: {value}", nameof(value));
                return Shape.Unranked;
            }

            var rank = (int)shape[0].FixedValue;
            return new Shape(Enumerable.Range(0, rank).Select(x => (Dimension)value[x]));
        }

        /// <summary>
        /// Get Prod.
        /// </summary>
        public Dimension Prod()
        {
            return _dimensions.Aggregate(new Dimension(1), (x, y) => x * y);
        }

        public FixedAndDynamicDimension ProdFixedAndDynamic()
        {
            var fixedValue = 1L;
            var dynamicValue = new Dimension(1);
            foreach (var dim in _dimensions)
            {
                if (dim.IsFixed)
                {
                    fixedValue *= dim.FixedValue;
                }
                else
                {
                    dynamicValue *= dim;
                }
            }

            return new(fixedValue, dynamicValue.IsFixed ? null : dynamicValue);
        }

        /// <summary>
        /// return new shape after insert dim.
        /// </summary>
        public Shape InsertAndClone(int index, Dimension dim)
        {
            var l = _dimensions.ToList();
            l.Insert(index, dim);
            return new Shape(l.ToArray());
        }

        /// <summary>
        /// return new shape after insert dim.
        /// </summary>
        public Shape InsertAndClone(int index, IEnumerable<Dimension> dims)
        {
            var l = _dimensions.ToList();
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
            return _dimensions.Select(dim => dim.FixedValue).ToList();
        }

        /// <summary>
        /// convert the int array.
        /// </summary>
        public long[] ToValueArray()
        {
            return _dimensions.Select(dim => dim.FixedValue).ToArray();
        }

        /// <inheritdoc/>
        public override string ToString() => Kind switch
        {
            ShapeKind.Invalid => "Invalid",
            ShapeKind.Unranked => "Unranked",
            _ => $"[{string.Join(',', _dimensions)}]",
        };

        /// <inheritdoc/>
        public int GetHashCode(IEqualityComparer comparer)
        {
            return ((IStructuralEquatable)_dimensions).GetHashCode(comparer);
        }

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            return _hashcode;
        }

        /// <inheritdoc/>
        public IEnumerator<Dimension> GetEnumerator()
        {
            return ((IEnumerable<Dimension>)_dimensions).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)_dimensions).GetEnumerator();
        }

        /// <inheritdoc/>
        public bool Equals(object? other, IEqualityComparer comparer)
        {
            return other is Shape shape && ((IStructuralEquatable)_dimensions).Equals(shape._dimensions, comparer);
        }

        /// <inheritdoc/>
        public bool Equals(Shape? other)
        {
            return other is not null && StructuralComparisons.StructuralEqualityComparer.Equals(_dimensions, other._dimensions);
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

            for (int i = 0; i < _dimensions.Length; i++)
            {
                if (!_dimensions[i].IsAssignableFrom(shape[i]))
                {
                    return false;
                }
            }

            return true;
        }

        public IR.Tuple ToTuple()
        {
            if (IsUnranked)
            {
                throw new InvalidOperationException("Cannot convert unranked shape to tuple");
            }

            return new IR.Tuple(_dimensions.Select(x => x.ToExpr()).ToArray());
        }

        private static ShapeKind KindOf(ReadOnlySpan<Dimension> dimensions)
        {
            return dimensions.AsValueEnumerable().Any(x => x.IsUnknown) ? ShapeKind.HasUnknownDimension : ShapeKind.Fixed;
        }
    }
}
