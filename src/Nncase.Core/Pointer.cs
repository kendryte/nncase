// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Pointer type.
/// </summary>
/// <typeparam name="T">Elem type.</typeparam>
public struct Pointer<T> : IEquatable<Pointer<T>>
    where T : unmanaged, IEquatable<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="Pointer{T}"/> struct.
    /// </summary>
    /// <param name="value">Value.</param>
    public Pointer(ulong value)
    {
        Value = value;
    }

    /// <summary>
    /// Gets element size.
    /// </summary>
    public static unsafe int ElemSize => sizeof(T);

    /// <summary>
    /// Gets value.
    /// </summary>
    public ulong Value { get; }

    /// <summary>
    /// Compare two pointers.
    /// </summary>
    /// <param name="left">Lhs.</param>
    /// <param name="right">Rhs.</param>
    /// <returns>Compare result.</returns>
    public static bool operator ==(Pointer<T> left, Pointer<T> right)
    {
        return left.Equals(right);
    }

    /// <summary>
    /// Compare two pointers.
    /// </summary>
    /// <param name="left">Lhs.</param>
    /// <param name="right">Rhs.</param>
    /// <returns>Compare result.</returns>
    public static bool operator !=(Pointer<T> left, Pointer<T> right)
    {
        return !(left == right);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return obj is Pointer<T> pointer && Equals(pointer);
    }

    /// <inheritdoc/>
    public bool Equals(Pointer<T> other)
    {
        return Value == other.Value;
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return Value.GetHashCode();
    }
}
