// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;

namespace Nncase.IR;

/// <summary>
/// the ir array
/// </summary>
/// <typeparam name="T"></typeparam>
public struct IRArray<T> : IStructuralEquatable, IEquatable<IRArray<T>>, IReadOnlyList<T>, IEnumerable<T>, IList<T>
{
    private int _hashcode;
    private readonly ImmutableArray<T> _array;

    /// <summary>
    /// check the ret
    /// </summary>
    public bool IsDefaultOrEmpty => _array.IsDefaultOrEmpty;

    /// <summary>
    /// construct Ir Array with array.
    /// </summary>
    /// <param name="array"></param>
    public IRArray(ImmutableArray<T> array)
    {
        _array = array;
        _hashcode = HashCode.Combine(StructuralComparisons.StructuralEqualityComparer.GetHashCode(_array));
    }

    /// <summary>
    /// ctor from ienumerable
    /// </summary>
    /// <param name="enumerable"></param>
    public IRArray(IEnumerable<T> enumerable) : this(enumerable.ToImmutableArray()) { }

    /// <summary>
    /// empty ir array
    /// </summary>
    public IRArray() : this(ImmutableArray<T>.Empty) { }

    /// <inheritdoc/>
    public T this[int index] => ((IReadOnlyList<T>)_array)[index];

    /// <inheritdoc/>
    public ReadOnlySpan<T> this[Range range] => _array.AsSpan()[range];

    T IList<T>.this[int index] { get => ((IList<T>)_array)[index]; set => ((IList<T>)_array)[index] = value; }
    /// <inheritdoc/>
    public int Count => ((IReadOnlyCollection<T>)_array).Count;
    /// <inheritdoc/>
    public bool IsReadOnly => ((ICollection<T>)_array).IsReadOnly;
    /// <inheritdoc/>
    public void Add(T item) { throw new InvalidOperationException("IRArray Can't Add Item!"); }
    /// <inheritdoc/>
    public void Clear() { throw new InvalidOperationException("IRArray Can't Clear Item!"); }
    /// <inheritdoc/>
    public bool Contains(T item)
    {
        return ((ICollection<T>)_array).Contains(item);
    }
    /// <inheritdoc/>
    public void CopyTo(T[] array, int arrayIndex)
    {
        ((ICollection<T>)_array).CopyTo(array, arrayIndex);
    }
    /// <inheritdoc/>
    public bool Equals(object? other, IEqualityComparer comparer)
    {
        return ((IStructuralEquatable)_array).Equals(other, comparer);
    }
    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return obj is IRArray<T> array && Equals(array);
    }
    /// <inheritdoc/>
    public bool Equals(IRArray<T> other)
    {
        return StructuralComparisons.StructuralEqualityComparer.Equals(_array, other._array);
    }
    /// <inheritdoc/>
    public IEnumerator<T> GetEnumerator()
    {
        return ((IEnumerable<T>)_array).GetEnumerator();
    }
    /// <inheritdoc/>
    public int GetHashCode(IEqualityComparer comparer)
    {
        return ((IStructuralEquatable)_array).GetHashCode(comparer);
    }
    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return _hashcode;
    }

    /// <inheritdoc/>
    public int IndexOf(T item)
    {
        return ((IList<T>)_array).IndexOf(item);
    }

    /// <inheritdoc/>
    public void Insert(int index, T item)
    {
        ((IList<T>)_array).Insert(index, item);
    }

    /// <inheritdoc/>
    public bool Remove(T item)
    {
        return ((ICollection<T>)_array).Remove(item);
    }

    /// <inheritdoc/>
    public void RemoveAt(int index)
    {
        ((IList<T>)_array).RemoveAt(index);
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return ((IEnumerable)_array).GetEnumerator();
    }

    /// <inheritdoc/>
    public static bool operator ==(IRArray<T> left, IRArray<T> right)
    {
        return left.Equals(right);
    }

    /// <inheritdoc/>
    public static bool operator !=(IRArray<T> left, IRArray<T> right)
    {
        return !(left == right);
    }

    /// <inheritdoc/>
    public static implicit operator IRArray<T>(ImmutableArray<T> array) =>
        new IRArray<T>(array);

    /// <inheritdoc/>
    public static implicit operator IRArray<T>(T[] array) =>
        new IRArray<T>(ImmutableArray.Create(array));
}
