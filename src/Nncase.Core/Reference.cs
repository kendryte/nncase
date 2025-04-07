// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Reference type.
/// </summary>
/// <typeparam name="T">Elem type.</typeparam>
public struct Reference<T> : IEquatable<Reference<T>>
    where T : IEquatable<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Reference{T}"/> struct.
    /// </summary>
    /// <param name="value">Value.</param>
    public Reference(T value)
    {
        Value = value;
    }

    /// <summary>
    /// Gets element size.
    /// </summary>
    public static unsafe int ElemSize => sizeof(ulong);

    /// <summary>
    /// Gets value.
    /// </summary>
    public T Value { get; }

    /// <summary>
    /// Compare two pointers.
    /// </summary>
    /// <param name="left">Lhs.</param>
    /// <param name="right">Rhs.</param>
    /// <returns>Compare result.</returns>
    public static bool operator ==(Reference<T> left, Reference<T> right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Compare two pointers.
    /// </summary>
    /// <param name="left">Lhs.</param>
    /// <param name="right">Rhs.</param>
    /// <returns>Compare result.</returns>
    public static bool operator !=(Reference<T> left, Reference<T> right)
    {
        return !(left == right);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return obj is Reference<T> pointer && Equals(pointer);
    }

    /// <inheritdoc/>
    public bool Equals(Reference<T> other)
    {
        return Value.Equals(other.Value);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return Value.GetHashCode();
    }
}

/// <summary>
/// Reference type.
/// </summary>
/// <param name="ElemType">type.</param>
public sealed record ReferenceType(DataType ElemType) : DataType
{
    /// <inheritdoc/>
    public override Type CLRType { get; } = typeof(Reference<>).MakeGenericType(ElemType.CLRType);

    /// <inheritdoc/>
    public override int SizeInBytes => sizeof(ulong);
}
