// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// BFloat16.
/// </summary>
public struct BFloat16 : IEquatable<BFloat16>, IComparable<BFloat16>, INumber<BFloat16>
{
    /// <summary>
    /// bfloat16 representation bits.
    /// </summary>
    public ushort _value;

    public static BFloat16 Infinity => FromRaw(0x7f80);

    public static BFloat16 NegInfinity => FromRaw(0xff80);

    public static BFloat16 MinValue => FromRaw(0xff7f);

    public static BFloat16 MaxValue => FromRaw(0x7f7f);

    public static BFloat16 Epsilon => FromRaw(0x3c00);

    public static BFloat16 NaN => FromRaw(0x7fc0);

    public static BFloat16 One => throw new NotImplementedException();

    public static int Radix => throw new NotImplementedException();

    public static BFloat16 Zero => throw new NotImplementedException();

    public static BFloat16 AdditiveIdentity => throw new NotImplementedException();

    public static BFloat16 MultiplicativeIdentity => throw new NotImplementedException();

    /// <summary>
    /// Implicit convert <see cref="BFloat16"/> to <see cref="float"/>.
    /// </summary>
    /// <param name="input">BFloat16 value.</param>
    public static implicit operator float(BFloat16 input)
    {
        float value;
        Unsafe.SkipInit(out value);
        Unsafe.As<float, int>(ref value) = input._value << 16;
        return value;
    }

    /// <summary>
    /// Explicit convert <see cref="float"/> to <see cref="BFloat16"/>.
    /// </summary>
    /// <param name="input">BFloat16 value.</param>
    public static explicit operator BFloat16(float input) => RoundToBFloat16(input);

    /// <summary>
    /// Compares values of two BFloat16 for binary equality.
    /// </summary>
    /// <param name="lhs">lhs.</param>
    /// <param name="rhs">rhls.</param>
    /// <returns>result of value comparisons.</returns>
    public static bool operator ==(BFloat16 lhs, BFloat16 rhs) => lhs._value == rhs._value;

    /// <summary>
    /// Compares values of two BFloat16 for binary inequality.
    /// </summary>
    /// <param name="lhs">lhs.</param>
    /// <param name="rhs">rhls.</param>
    /// <returns>result of value comparisons.</returns>
    public static bool operator !=(BFloat16 lhs, BFloat16 rhs) => lhs._value != rhs._value;

    public static bool operator <(BFloat16 left, BFloat16 right)
    {
        return left.CompareTo(right) < 0;
    }

    public static bool operator <=(BFloat16 left, BFloat16 right)
    {
        return left.CompareTo(right) <= 0;
    }

    public static bool operator >(BFloat16 left, BFloat16 right)
    {
        return left.CompareTo(right) > 0;
    }

    public static bool operator >=(BFloat16 left, BFloat16 right)
    {
        return left.CompareTo(right) >= 0;
    }

    public static BFloat16 operator %(BFloat16 left, BFloat16 right) => (BFloat16)((float)left % (float)right);

    public static BFloat16 operator +(BFloat16 left, BFloat16 right) => (BFloat16)((float)left + (float)right);

    public static BFloat16 operator --(BFloat16 value) => (BFloat16)((float)value - 1);

    public static BFloat16 operator /(BFloat16 left, BFloat16 right) => (BFloat16)((float)left / (float)right);

    public static BFloat16 operator ++(BFloat16 value) => (BFloat16)((float)value + 1);

    public static BFloat16 operator *(BFloat16 left, BFloat16 right) => (BFloat16)((float)left * (float)right);

    public static BFloat16 operator -(BFloat16 left, BFloat16 right) => (BFloat16)((float)left - (float)right);

    public static BFloat16 operator -(BFloat16 value) => (BFloat16)(-(float)value);

    public static BFloat16 operator +(BFloat16 value) => (BFloat16)(+(float)value);

    /// <summary>
    /// Reinterpret cast <see cref="ushort"/> to <see cref="BFloat16"/>.
    /// </summary>
    /// <param name="value">UInt16 value.</param>
    /// <returns>Casted bfloat16.</returns>
    public static BFloat16 FromRaw(ushort value)
    {
        BFloat16 result;
        Unsafe.SkipInit(out result);
        result._value = value;
        return result;
    }

    /// <summary>
    /// Convert <see cref="float"/> to <see cref="BFloat16"/> using rounding.
    /// </summary>
    /// <param name="value">Float value.</param>
    /// <returns>Converted bfloat16.</returns>
    public static BFloat16 RoundToBFloat16(float value)
    {
        if (float.IsNaN(value))
        {
            // If the value is a NaN, squash it to a qNaN with msb of fraction set,
            // this makes sure after truncation we don't end up with an inf.
            //
            // qNaN magic: All exponent bits set + most significant bit of fraction
            // set.
            return FromRaw(0x7fc0);
        }

        var input = Unsafe.As<float, uint>(ref value);

        // Least significant bit of resulting bfloat.
        uint lsb = (input >> 16) & 1;
        uint roundingBias = 0x7fff + lsb;
        input += roundingBias;
        return FromRaw((ushort)(input >> 16));
    }

    public static BFloat16 Abs(BFloat16 value) => throw new NotImplementedException();

    public static bool IsCanonical(BFloat16 value) => throw new NotImplementedException();

    public static bool IsComplexNumber(BFloat16 value) => throw new NotImplementedException();

    public static bool IsEvenInteger(BFloat16 value) => throw new NotImplementedException();

    public static bool IsFinite(BFloat16 value) => throw new NotImplementedException();

    public static bool IsImaginaryNumber(BFloat16 value) => throw new NotImplementedException();

    public static bool IsInfinity(BFloat16 value) => float.IsInfinity(value);

    public static bool IsInteger(BFloat16 value) => throw new NotImplementedException();

    public static bool IsNaN(BFloat16 value) => throw new NotImplementedException();

    public static bool IsNegative(BFloat16 value) => throw new NotImplementedException();

    public static bool IsNegativeInfinity(BFloat16 value) => float.IsNegativeInfinity(value);

    public static bool IsNormal(BFloat16 value) => throw new NotImplementedException();

    public static bool IsOddInteger(BFloat16 value) => throw new NotImplementedException();

    public static bool IsPositive(BFloat16 value) => throw new NotImplementedException();

    public static bool IsPositiveInfinity(BFloat16 value) => throw new NotImplementedException();

    public static bool IsRealNumber(BFloat16 value) => throw new NotImplementedException();

    public static bool IsSubnormal(BFloat16 value) => throw new NotImplementedException();

    public static bool IsZero(BFloat16 value) => throw new NotImplementedException();

    public static BFloat16 MaxMagnitude(BFloat16 x, BFloat16 y) => throw new NotImplementedException();

    public static BFloat16 MaxMagnitudeNumber(BFloat16 x, BFloat16 y) => throw new NotImplementedException();

    public static BFloat16 MinMagnitude(BFloat16 x, BFloat16 y) => throw new NotImplementedException();

    public static BFloat16 MinMagnitudeNumber(BFloat16 x, BFloat16 y) => throw new NotImplementedException();

    public static BFloat16 Parse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider) => throw new NotImplementedException();

    public static BFloat16 Parse(string s, NumberStyles style, IFormatProvider? provider) => throw new NotImplementedException();

    public static bool TryConvertFromChecked<TOther>(TOther value, [MaybeNullWhen(false)] out BFloat16 result)
        where TOther : INumberBase<TOther>
        => throw new NotImplementedException();

    public static bool TryConvertFromSaturating<TOther>(TOther value, [MaybeNullWhen(false)] out BFloat16 result)
        where TOther : INumberBase<TOther>
        => throw new NotImplementedException();

    public static bool TryConvertFromTruncating<TOther>(TOther value, [MaybeNullWhen(false)] out BFloat16 result)
        where TOther : INumberBase<TOther>
        => throw new NotImplementedException();

    public static bool TryConvertToChecked<TOther>(BFloat16 value, [MaybeNullWhen(false)] out TOther result)
        where TOther : INumberBase<TOther>
        => throw new NotImplementedException();

    public static bool TryConvertToSaturating<TOther>(BFloat16 value, [MaybeNullWhen(false)] out TOther result)
        where TOther : INumberBase<TOther>
        => throw new NotImplementedException();

    public static bool TryConvertToTruncating<TOther>(BFloat16 value, [MaybeNullWhen(false)] out TOther result)
        where TOther : INumberBase<TOther>
        => throw new NotImplementedException();

    public static bool TryParse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider, [MaybeNullWhen(false)] out BFloat16 result) => throw new NotImplementedException();

    public static bool TryParse([NotNullWhen(true)] string? s, NumberStyles style, IFormatProvider? provider, [MaybeNullWhen(false)] out BFloat16 result) => throw new NotImplementedException();

    public static BFloat16 Parse(ReadOnlySpan<char> s, IFormatProvider? provider) => throw new NotImplementedException();

    public static bool TryParse(ReadOnlySpan<char> s, IFormatProvider? provider, [MaybeNullWhen(false)] out BFloat16 result) => throw new NotImplementedException();

    public static BFloat16 Parse(string s, IFormatProvider? provider) => throw new NotImplementedException();

    public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider, [MaybeNullWhen(false)] out BFloat16 result) => throw new NotImplementedException();

    /// <summary>
    /// Returns a value indicating whether this instance and other BFloat16 represent the same value.
    /// </summary>
    /// <param name="other">A BFloat16 object to compare to this instance.</param>
    /// <returns>true if other.value is equal to this instance; otherwise, false.</returns>
    public bool Equals(BFloat16 other)
    {
        return other == this;
    }

    /// <summary>
    /// Returns a value indicating whether this instance and a specified System.Object
    /// represent the same type and value.
    /// </summary>
    /// <param name="obj">An System.Object.</param>
    /// <returns>true if obj is BFloat16 its value is equal to this instance; otherwise, false.</returns>
    public override bool Equals(object? obj)
    {
        bool result = false;
        if (obj is BFloat16)
        {
            var bfl16 = (BFloat16)obj;
            result = bfl16 == this;
        }

        return result;
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
    public override string ToString()
    {
        return ((float)this).ToString();
    }

    /// <inheritdoc/>
    public int CompareTo(BFloat16 other)
    {
        return ((float)this).CompareTo(other);
    }

    public int CompareTo(object? obj) => throw new NotImplementedException();

    public bool TryFormat(Span<char> destination, out int charsWritten, ReadOnlySpan<char> format, IFormatProvider? provider) => throw new NotImplementedException();

    public string ToString(string? format, IFormatProvider? formatProvider) => throw new NotImplementedException();
}