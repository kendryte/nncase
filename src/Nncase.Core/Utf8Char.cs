// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Utf8 char.
/// </summary>
public struct Utf8Char : IEquatable<Utf8Char>, IComparable<Utf8Char>
{
    private byte _value;

    /// <summary>
    /// Implicit convert <see cref="Utf8Char"/> to <see cref="byte"/>.
    /// </summary>
    /// <param name="char">Utf8 char.</param>
    public static implicit operator byte(Utf8Char @char) => @char._value;

    /// <summary>
    /// Implicit convert <see cref="byte"/> to <see cref="Utf8Char"/>.
    /// </summary>
    /// <param name="byte">Byte.</param>
    public static implicit operator Utf8Char(byte @byte)
    {
        Utf8Char @char;
        Unsafe.SkipInit(out @char);
        @char._value = @byte;
        return @char;
    }

    /// <summary>
    /// Compares values of two Utf8Char for binary equality.
    /// </summary>
    /// <param name="lhs">lhs.</param>
    /// <param name="rhs">rhs.</param>
    /// <returns>result of value comparisons.</returns>
    public static bool operator ==(Utf8Char lhs, Utf8Char rhs) => lhs._value == rhs._value;

    /// <summary>
    /// Compares values of two Utf8Char for binary inequality.
    /// </summary>
    /// <param name="lhs">lhs.</param>
    /// <param name="rhs">rhs.</param>
    /// <returns>result of value comparisons.</returns>
    public static bool operator !=(Utf8Char lhs, Utf8Char rhs) => lhs._value != rhs._value;

    /// <summary>
    /// Compares whether lhs less than rhs.
    /// </summary>
    /// <param name="lhs">lhs.</param>
    /// <param name="rhs">rhs.</param>
    /// <returns>result of value comparisons.</returns>
    public static bool operator <(Utf8Char lhs, Utf8Char rhs)
    {
        return lhs.CompareTo(rhs) < 0;
    }

    /// <summary>
    /// Compares whether lhs less than or equal to rhs.
    /// </summary>
    /// <param name="lhs">lhs.</param>
    /// <param name="rhs">rhs.</param>
    /// <returns>result of value comparisons.</returns>
    public static bool operator <=(Utf8Char lhs, Utf8Char rhs)
    {
        return lhs.CompareTo(rhs) <= 0;
    }

    /// <summary>
    /// Compares whether lhs greater than rhs.
    /// </summary>
    /// <param name="lhs">lhs.</param>
    /// <param name="rhs">rhs.</param>
    /// <returns>result of value comparisons.</returns>
    public static bool operator >(Utf8Char lhs, Utf8Char rhs)
    {
        return lhs.CompareTo(rhs) > 0;
    }

    /// <summary>
    /// Compares whether lhs greater than or equal to rhs.
    /// </summary>
    /// <param name="lhs">lhs.</param>
    /// <param name="rhs">rhs.</param>
    /// <returns>result of value comparisons.</returns>
    public static bool operator >=(Utf8Char lhs, Utf8Char rhs)
    {
        return lhs.CompareTo(rhs) >= 0;
    }

    /// <inheritdoc/>
    public bool Equals(Utf8Char other)
    {
        return _value == other._value;
    }

    /// <summary>
    /// Returns a value indicating whether this instance and a specified System.Object
    /// represent the same type and value.
    /// </summary>
    /// <param name="obj">An System.Object.</param>
    /// <returns>true if obj is Utf8Char its value is equal to this instance; otherwise, false.</returns>
    public override bool Equals(object? obj)
    {
        if (obj is Utf8Char)
        {
            return (Utf8Char)obj == this;
        }

        return false;
    }

    /// <summary>
    /// Returns the hash code for this instance.
    /// </summary>
    /// <returns>A 32-bit signed integer hash code.</returns>
    public override int GetHashCode()
    {
        return _value.GetHashCode();
    }

    /// <inheritdoc/>
    public int CompareTo(Utf8Char other)
    {
        return _value.CompareTo(other._value);
    }
}
