// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace Nncase.IR
{
    public struct IRArray<T> : IStructuralEquatable, IEquatable<IRArray<T>>, IReadOnlyList<T>, IEnumerable<T>, IList<T>
    {
        private int _hashcode;
        private readonly ImmutableArray<T> _array;

        /// <summary>
        /// construct Ir Array with array.
        /// </summary>
        /// <param name="array"></param>
        public IRArray(ImmutableArray<T> array)
        {
            _array = array;
            _hashcode = HashCode.Combine(StructuralComparisons.StructuralEqualityComparer.GetHashCode(_array));
        }

        public IRArray(IEnumerable<T> enumerable) : this(enumerable.ToImmutableArray()) { }

        public T this[int index] => ((IReadOnlyList<T>)_array)[index];

        T IList<T>.this[int index] { get => ((IList<T>)_array)[index]; set => ((IList<T>)_array)[index] = value; }

        public int Count => ((IReadOnlyCollection<T>)_array).Count;

        public bool IsReadOnly => ((ICollection<T>)_array).IsReadOnly;

        public void Add(T item)
        {
            ((ICollection<T>)_array).Add(item);
        }

        public void Clear()
        {
            ((ICollection<T>)_array).Clear();
        }

        public bool Contains(T item)
        {
            return ((ICollection<T>)_array).Contains(item);
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            ((ICollection<T>)_array).CopyTo(array, arrayIndex);
        }

        public bool Equals(object? other, IEqualityComparer comparer)
        {
            return ((IStructuralEquatable)_array).Equals(other, comparer);
        }

        public override bool Equals(object? obj)
        {
            return obj is IRArray<T> array && Equals(array);
        }

        public bool Equals(IRArray<T> other)
        {
            return StructuralComparisons.StructuralEqualityComparer.Equals(_array, other._array);
        }

        public IEnumerator<T> GetEnumerator()
        {
            return ((IEnumerable<T>)_array).GetEnumerator();
        }

        public int GetHashCode(IEqualityComparer comparer)
        {
            return ((IStructuralEquatable)_array).GetHashCode(comparer);
        }

        public override int GetHashCode()
        {
            return _hashcode;
        }

        public int IndexOf(T item)
        {
            return ((IList<T>)_array).IndexOf(item);
        }

        public void Insert(int index, T item)
        {
            ((IList<T>)_array).Insert(index, item);
        }

        public bool Remove(T item)
        {
            return ((ICollection<T>)_array).Remove(item);
        }

        public void RemoveAt(int index)
        {
            ((IList<T>)_array).RemoveAt(index);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)_array).GetEnumerator();
        }

        public static bool operator ==(IRArray<T> left, IRArray<T> right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(IRArray<T> left, IRArray<T> right)
        {
            return !(left == right);
        }

        public static implicit operator IRArray<T>(ImmutableArray<T> array) =>
            new IRArray<T>(array);

        public static implicit operator IRArray<T>(T[] array) =>
            new IRArray<T>(ImmutableArray.Create(array));
    }
}