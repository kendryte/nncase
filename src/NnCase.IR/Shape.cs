using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR
{
    public struct Shape : IEquatable<Shape>
    {
        public const int MaxSmallSize = 4;

        private unsafe fixed int _smallValues[MaxSmallSize];
        private readonly int[] _largeValues;

        public int this[int index]
        {
            get
            {
                if (index >= Count)
                    throw new ArgumentOutOfRangeException(nameof(index));

                if (Count <= MaxSmallSize)
                {
                    unsafe
                    {
                        return _smallValues[index];
                    }
                }
                else
                {
                    return _largeValues[index];
                }
            }
        }

        public int Count { get; }

        public unsafe Shape(ReadOnlySpan<int> shape)
        {
            Count = shape.Length;
            if (Count <= MaxSmallSize)
            {
                for (int i = 0; i < Count; i++)
                    _smallValues[i] = shape[i];
                _largeValues = null;
            }
            else
            {
                _largeValues = shape.ToArray();
            }
        }

        public static unsafe implicit operator ReadOnlySpan<int>(Shape shape)
        {
            if (shape.Count <= MaxSmallSize)
            {
                return new ReadOnlySpan<int>(shape._smallValues, shape.Count);
            }
            else
            {
                return shape._largeValues;
            }
        }

        public Enumerator GetEnumerator()
            => new Enumerator(this);

        public unsafe int[] ToArray()
        {
            if (Count <= MaxSmallSize)
            {
                var array = new int[Count];

                for (int i = 0; i < Count; i++)
                    array[i] = _smallValues[i];
                return array;
            }
            else
            {
                return (int[])_largeValues.Clone();
            }
        }

        public override string ToString()
        {
            return $"({string.Join(",", ToArray())})";
        }

        public override bool Equals(object obj)
        {
            return obj is Shape shape && Equals(shape);
        }

        public unsafe bool Equals(Shape other)
        {
            if (Count == other.Count)
            {
                if (Count <= MaxSmallSize)
                {
                    for (int i = 0; i < Count; i++)
                    {
                        if (_smallValues[i] != other._smallValues[i])
                            return false;
                    }
                }
                else
                {
                    return Array.Equals(_largeValues, other._largeValues);
                }

                return true;
            }

            return false;
        }

        public unsafe override int GetHashCode()
        {
            int hashCode = 0;

            if (Count <= MaxSmallSize)
            {
                for (int i = 0; i < Count; i++)
                {
                    hashCode = HashCode.Combine(hashCode, _smallValues[i].GetHashCode());
                }
            }
            else
            {
                for (int i = 0; i < Count; i++)
                {
                    hashCode = HashCode.Combine(hashCode, _largeValues[i].GetHashCode());
                }
            }

            return hashCode;
        }

        public static bool operator ==(Shape left, Shape right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Shape left, Shape right)
        {
            return !(left == right);
        }

        public ref struct Enumerator
        {
            private ReadOnlySpan<int> _values;
            private int _index;

            public int Current => _values[_index];

            internal Enumerator(ReadOnlySpan<int> values)
            {
                _values = values;
                _index = -1;
            }

            public bool MoveNext()
            {
                var index = _index + 1;
                if (index < _values.Length)
                {
                    _index = index;
                    return true;
                }

                return false;
            }
        }
    }
}
