// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace Nncase;

public interface IReference
{
    object Value { get; }
}

/// <summary>
/// Reference type.
/// </summary>
/// <typeparam name="T">Elem type.</typeparam>
[JsonConverter(typeof(IO.ReferenceJsonConverter))]
public struct Reference<T> : IReference, IEquatable<Reference<T>>
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

    object IReference.Value => Value!;

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
        return EqualityComparer<T>.Default.Equals(Value, other.Value);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return HashCode.Combine(Value);
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
    public override int SizeInBytes => ElemType.SizeInBytes; // todo set at 8.
}
