// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
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
    public sealed class Shape : IReadOnlyList<Dimension>, IEquatable<Shape?>
    {
        private ReadOnlyCollection<Dimension> _dimensions;

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(IEnumerable<Dimension> dimensions)
        {
            Kind = KindOf(dimensions);
            _dimensions = dimensions.ToList().AsReadOnly();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(IEnumerable<long> dimensions)
        {
            Kind = ShapeKind.Fixed;
            _dimensions = dimensions.Select(x => new Dimension((int)x)).ToList().AsReadOnly();
        }

        public static implicit operator Shape(int[] dimensions) => new Shape((ReadOnlySpan<int>)dimensions);

        public static bool operator ==(Shape? left, Shape? right)
        {
            return EqualityComparer<Shape>.Default.Equals(left, right);
        }

        public static bool operator !=(Shape? left, Shape? right)
        {
            return !(left == right);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(ReadOnlySpan<long> dimensions)
        {
            Kind = ShapeKind.Fixed;
            _dimensions = dimensions.AsValueEnumerable().Select(x => new Dimension((int)x)).ToList().AsReadOnly();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="dimensions">Dimensions.</param>
        public Shape(ReadOnlySpan<int> dimensions)
        {
            Kind = ShapeKind.Fixed;
            _dimensions = dimensions.AsValueEnumerable().Select(x => new Dimension(x)).ToList().AsReadOnly();
        }

        private Shape(ShapeKind kind, IEnumerable<Dimension> dimensions)
        {
            Kind = kind;
            _dimensions = dimensions.ToList().AsReadOnly();
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
        public bool IsScalar => IsFixed && _dimensions.Count == 0;

        /// <summary>
        /// Gets rank.
        /// </summary>
        public int Rank => _dimensions.Count;

        public int Size => Enumerable.Range(0, Rank).Aggregate(1, (size, i) => size * _dimensions[i].FixedValue);

        int IReadOnlyCollection<Dimension>.Count => Rank;

        /// <inheritdoc/>
        public Dimension this[int index] => _dimensions[index];

        /// <summary>
        /// Searches for the specified item and returns the zero-based index of the first occurrence.
        /// </summary>
        /// <param name="item">The item to find.</param>
        /// <returns>The founded index or -1.</returns>
        public int IndexOf(Dimension item) => _dimensions.IndexOf(item);

        /// <summary>
        /// Determine whether the item is in the <seealso cref="Shape"/>.
        /// </summary>
        /// <param name="item">The item to find.</param>
        /// <returns>true if value is found, otherwise false.</returns>
        public bool Contains(Dimension item) => _dimensions.Contains(item);

        /// <summary>
        /// Copy all dimensions to an array.
        /// </summary>
        /// <param name="array">The desitination array.</param>
        /// <param name="arrayIndex">The zero-based index in array at which copying begins.</param>
        public void CopyTo(Dimension[] array, int arrayIndex) => _dimensions.CopyTo(array, arrayIndex);

        /// <summary>
        /// Get enumerator.
        /// </summary>
        /// <returns>The enumerator.</returns>
        public IEnumerator<Dimension> GetEnumerator() => _dimensions.GetEnumerator();

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"[{string.Join(',', _dimensions)}]";
        }

        IEnumerator<Dimension> IEnumerable<Dimension>.GetEnumerator() => GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        private static ShapeKind KindOf(IEnumerable<Dimension> dimensions)
        {
            return dimensions.Any(x => x.IsUnknown) ? ShapeKind.HasUnknownDimension : ShapeKind.Fixed;
        }

        public override bool Equals(object? obj)
        {
            return Equals(obj as Shape);
        }

        public bool Equals(Shape? other)
        {
            if (other is null)
            {
                return false;
            }
            return _dimensions.SequenceEqual(other._dimensions);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(StructuralComparisons.StructuralEqualityComparer.GetHashCode(_dimensions));
        }

        public static implicit operator ReadOnlySpan<int>(Shape shape) => shape._dimensions.Select(x => (int)(x.Value ?? -1)).ToArray();

    }
}
