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

    /// <summary>
    /// Tensor shape.
    /// </summary>
    public sealed class Shape : IStructuralEquatable, IReadOnlyList<Dimension>, IEquatable<Shape>, IEnumerable<Dimension>
    {
        private readonly ImmutableArray<Dimension> _dimensions;

        private int _hashcode;

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(IEnumerable<Dimension> dimensions)
        {
            Kind = KindOf(dimensions);
            _dimensions = dimensions.ToImmutableArray();
            _hashcode = HashCode.Combine(StructuralComparisons.StructuralEqualityComparer.GetHashCode(_dimensions));
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(IEnumerable<long> dimensions)
        {
            Kind = ShapeKind.Fixed;
            _dimensions = dimensions.Select(x => new Dimension((int)x)).ToImmutableArray();
            _hashcode = HashCode.Combine(StructuralComparisons.StructuralEqualityComparer.GetHashCode(_dimensions));
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(IEnumerable<int> dimensions)
        {
            Kind = ShapeKind.Fixed;
            _dimensions = dimensions.AsValueEnumerable().Select(x => new Dimension(x)).ToImmutableArray();
            _hashcode = HashCode.Combine(StructuralComparisons.StructuralEqualityComparer.GetHashCode(_dimensions));
        }

        /// <summary>
        /// init from the dimensions
        /// </summary>
        /// <param name="dimensions"></param>
        public Shape(params Dimension[] dimensions)
            : this(dimensions.AsValueEnumerable())
        { }

        private Shape(ShapeKind kind, IEnumerable<Dimension> dimensions)
        {
            Kind = kind;
            _dimensions = dimensions.ToImmutableArray();
            _hashcode = HashCode.Combine(StructuralComparisons.StructuralEqualityComparer.GetHashCode(_dimensions));
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
        /// Get Total Elements
        /// </summary>
        public int Size => Enumerable.Range(0, Rank).Aggregate(1, (size, i) => size * _dimensions[i].FixedValue);

        /// <inheritdoc/>
        public int Count => ((IReadOnlyCollection<Dimension>)_dimensions).Count;

        /// <inheritdoc/>
        public Dimension this[int index] =>
            index >= 0
                ? ((IReadOnlyList<Dimension>)_dimensions)[index]
                : ((IReadOnlyList<Dimension>)_dimensions)[Rank + index];

        /// <summary>
        /// Get Pord
        /// </summary>
        /// <returns></returns>
        public Dimension Prod()
        {
            return _dimensions.Aggregate(new Dimension(1), (x, y) => x * y);
        }
        /// <summary>
        /// return new shape after insert dim
        /// </summary>
        /// <param name="index"></param>
        /// <param name="dim"></param>
        /// <returns></returns>
        public Shape InsertAndClone(int index, Dimension dim)
        {
            var l = _dimensions.ToList();
            l.Insert(index, dim);
            return new Shape(l);
        }

        /// <summary>
        /// return new shape after insert dim
        /// </summary>
        /// <param name="index"></param>
        /// <param name="dims"></param>
        /// <returns></returns>
        public Shape InsertAndClone(int index, IEnumerable<Dimension> dims)
        {
            var l = _dimensions.ToList();
            foreach (var d in dims)
            {
                l.Insert(index++, d);
            }
            return new Shape(l);
        }

        /// <summary>
        /// convert to the int list
        /// </summary>
        /// <returns></returns>
        public List<int> ToValueList()
        {
            return _dimensions.Select(dim => dim.FixedValue).ToList();
        }

        /// <summary>
        /// convert the int array
        /// </summary>
        /// <returns></returns>
        public int[] ToValueArray()
        {
            return _dimensions.Select(dim => dim.FixedValue).ToArray();
        }

        /// <inheritdoc/>
        public override string ToString() => Kind switch
        {
            ShapeKind.Invalid => "Invalid",
            ShapeKind.Unranked => "Unranked",
            _ => $"[{string.Join(',', _dimensions)}]"
        };

        private static ShapeKind KindOf(IEnumerable<Dimension> dimensions)
        {
            return dimensions.Any(x => x.IsUnknown) ? ShapeKind.HasUnknownDimension : ShapeKind.Fixed;
        }
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
            return ((IStructuralEquatable)_dimensions).Equals(other, comparer);
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
        
        /// <inheritdoc/>
        public static implicit operator ReadOnlySpan<int>(Shape shape) => shape._dimensions.Select(x => (int)(x.Value ?? -1)).ToArray();

        /// <inheritdoc/>
        public static bool operator ==(Shape lhs, Shape rhs) { return lhs.Equals(rhs); }

        /// <inheritdoc/>
        public static bool operator !=(Shape lhs, Shape rhs) { return !(lhs == rhs); }

        /// <inheritdoc/>
        public static implicit operator Shape(Dimension[] dimensions) => new Shape(dimensions);
        
        /// <inheritdoc/>
        public static implicit operator Shape(int[] dimensions) => new Shape(dimensions);
    }
}
