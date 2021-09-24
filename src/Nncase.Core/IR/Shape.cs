using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Tensor shape.
    /// </summary>
    public sealed class Shape : IList<Dimension>, IReadOnlyList<Dimension>
    {
        private ShapeKind _kind;
        private List<Dimension> _dimensions;

        private Shape(ShapeKind kind, List<Dimension> dimensions, bool isReadOnly = false)
        {
            _kind = kind;
            _dimensions = dimensions;
            IsReadOnly = isReadOnly;
        }

        private enum ShapeKind
        {
            Invalid,
            Unranked,
            HasUnknownDimension,
            Fixed,
        }

        /// <summary>
        /// Gets an invalid shape.
        /// </summary>
        public static Shape Invalid { get; } = new Shape(ShapeKind.Invalid, new List<Dimension>(), isReadOnly: true);

        /// <summary>
        /// Gets an unranked shape.
        /// </summary>
        public static Shape Unranked { get; } = new Shape(ShapeKind.Unranked, new List<Dimension>(), isReadOnly: true);

        /// <summary>
        /// Gets a scalar shape.
        /// </summary>
        public static Shape Scalar { get; } = new Shape(ShapeKind.Fixed, new List<Dimension>(), isReadOnly: true);

        /// <summary>
        /// Gets a value indicating whether is readonly.
        /// </summary>
        public bool IsReadOnly { get; }

        /// <inheritdoc/>
        public int Count => _dimensions.Count;

        /// <inheritdoc/>
        public Dimension this[int index] { get => _dimensions[index]; set => throw new NotImplementedException(); }

        /// <inheritdoc/>
        public int IndexOf(Dimension item) => _dimensions.IndexOf(item);

        public void Insert(int index, Dimension item)
        {
            throw new NotImplementedException();
        }

        public void RemoveAt(int index)
        {
            throw new NotImplementedException();
        }

        public void Add(Dimension item)
        {
            throw new NotImplementedException();
        }

        public void Clear()
        {
            throw new NotImplementedException();
        }

        public bool Contains(Dimension item)
        {
            throw new NotImplementedException();
        }

        public void CopyTo(Dimension[] array, int arrayIndex)
        {
            throw new NotImplementedException();
        }

        public bool Remove(Dimension item)
        {
            throw new NotImplementedException();
        }

        public IEnumerator<Dimension> GetEnumerator()
        {
            throw new NotImplementedException();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            throw new NotImplementedException();
        }
    }
}
