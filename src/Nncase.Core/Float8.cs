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
/// Float8E4M3.
/// </summary>
public struct Float8E4M3 : IEquatable<Float8E4M3>, IComparable<Float8E4M3>, INumber<Float8E4M3>
{
    /// <summary>
    /// FP8 E4M3 representation bits.
    /// </summary>
    public byte _value;

    public static Float8E4M3 NaN => FromRaw(0b1111111);

    public static Float8E4M3 Infinity => NaN;

    public static Float8E4M3 NegInfinity => NaN;

    public static Float8E4M3 Zero => FromRaw(0b0000000);

    public static Float8E4M3 MaxNormal => FromRaw(0b1111110);

    public static Float8E4M3 MinNormal => FromRaw(0b0010000);

    public static Float8E4M3 MaxSubnormal => FromRaw(0b0000111);

    public static Float8E4M3 MinSubnormal => FromRaw(0b0000001);

    public static Float8E4M3 One => throw new NotImplementedException();

    public static int Radix => throw new NotImplementedException();

    public static Float8E4M3 AdditiveIdentity => throw new NotImplementedException();

    public static Float8E4M3 MultiplicativeIdentity => throw new NotImplementedException();

    static Float8E4M3 INumberBase<Float8E4M3>.One => throw new NotImplementedException();

    static int INumberBase<Float8E4M3>.Radix => throw new NotImplementedException();

    static Float8E4M3 INumberBase<Float8E4M3>.Zero => throw new NotImplementedException();

    static Float8E4M3 IAdditiveIdentity<Float8E4M3, Float8E4M3>.AdditiveIdentity => throw new NotImplementedException();

    static Float8E4M3 IMultiplicativeIdentity<Float8E4M3, Float8E4M3>.MultiplicativeIdentity => throw new NotImplementedException();

    /// <summary>
    /// Implicit conversion from Float8E4M3 to float.
    /// </summary>
    /// <param name="input">FP8 value.</param>
    public static implicit operator float(Float8E4M3 input)
    {
        const bool IS_E4M3 = true;

        // Number of Bits representing mantissa and exponents
        const int FP32_NUM_BITS = 32;
        const int FP32_NUM_MANTISSA_BITS = 23;
        const int FP32_EXPONENT_BIAS = 127;

        const int FP8_NUM_BITS = 8;
        const int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;
        const int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;
        const int FP8_EXPONENT_BIAS = IS_E4M3 ? 7 : 15;
        const byte FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1;
        const byte FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1;

        const uint kF32_NaN = 0x7fffffff;

        byte f8 = input._value;
        int sign = (f8 >> (FP8_NUM_BITS - 1)) & 1;
        int exp = (f8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK;
        int mantissa = f8 & FP8_MANTISSA_MASK;
        uint f = (uint)(sign << (FP32_NUM_BITS - 1));

        if (IS_E4M3 && exp == 15 && mantissa == 0x7)
        {
            // Handle special case for NaN in E4M3 format
            f = kF32_NaN;
        }
        else if (exp > 0 && IS_E4M3)
        {
            // Normal case
            exp += FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS;
            f = (uint)(f | ((uint)exp << FP32_NUM_MANTISSA_BITS) | ((uint)mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
        }
        else if (exp == 0)
        {
            if (mantissa != 0)
            {
                // Subnormal case
                exp += FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS + 1;
                while ((mantissa & (1 << FP8_NUM_MANTISSA_BITS)) == 0)
                {
                    mantissa <<= 1;
                    exp--;
                }

                mantissa &= FP8_MANTISSA_MASK;
                f = (uint)(f | ((uint)exp << FP32_NUM_MANTISSA_BITS) | ((uint)mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
            }

            // Sign-preserving zero is already handled by f being zeroed
        }
        else
        {
            if (mantissa == 0)
            {
                // Sign-preserving infinity
                f |= 0x7f800000;
            }
            else
            {
                // Canonical NaN
                f = kF32_NaN;
            }
        }

        // Return as float
        return BitConverter.ToSingle(BitConverter.GetBytes(f), 0);
    }

    /// <summary>
    /// Explicit conversion from float to Float8E4M3.
    /// </summary>
    /// <param name="input">Float value.</param>
    public static explicit operator Float8E4M3(float input)
    {
        const bool IS_E4M3 = true;

        // Number of Bits representing mantissa and exponents
        const int FP32_NUM_BITS = 32;
        const int FP32_NUM_MANTISSA_BITS = 23;
        const int FP32_EXPONENT_BIAS = 127;

        const int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;
        const int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;
        const int FP8_MAX_EXPONENT = IS_E4M3 ? 7 : 15;
        const int FP8_MIN_EXPONENT = IS_E4M3 ? -6 : -14;
        const int FP8_EXPONENT_BIAS = IS_E4M3 ? 7 : 15;

        const byte FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1;
        const byte FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1;

        const byte FP8_MAX_FLT = IS_E4M3 ? 0x7e : 0x7b;

        // Extract float bits using BitConverter
        uint s = BitConverter.ToUInt32(BitConverter.GetBytes(input), 0);

        // Extract sign, exponent and mantissa
        byte sign = (byte)((s >> 24) & 0x80);
        int exp = (int)(((s >> FP32_NUM_MANTISSA_BITS) & 0xff) - FP32_EXPONENT_BIAS);
        int mantissa = (int)(s & 0x7fffff);
        const byte kF8_NaN = 0x7f;

        // NaN => NaN
        if (float.IsNaN(input))
        {
            return FromRaw(kF8_NaN);
        }

        // Inf => MAX_FLT (satfinite)
        if (float.IsInfinity(input))
        {
            return FromRaw((byte)(sign | FP8_MAX_FLT));
        }

        // Special handling for exponent -128
        if (exp == -128)
        {
            return FromRaw((byte)(sign | FP8_MAX_FLT));
        }

        int sticky_bit = 0;
        bool skip_sign = false;
        bool may_be_nan = false;

        byte u;
        if (exp >= FP8_MIN_EXPONENT && exp <= FP8_MAX_EXPONENT)
        {
            // Normal fp32 to normal fp8
            exp = (int)(exp + FP8_EXPONENT_BIAS);
            u = (byte)((exp & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS);
            u = (byte)(u | (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
        }
        else if (exp < FP8_MIN_EXPONENT)
        {
            // fp32 to subnormal fp8
            int rshift = FP8_MIN_EXPONENT - exp;
            if (rshift < FP32_NUM_BITS)
            {
                mantissa |= 1 << FP32_NUM_MANTISSA_BITS;
                sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0) ? 1 : 0;
                mantissa = mantissa >> rshift;
                u = (byte)((mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK);
            }
            else
            {
                mantissa = 0;
                u = 0;
            }
        }
        else if (exp == FP8_MAX_EXPONENT + 1)
        {
            byte mantissaTmp = (byte)(mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
            if (mantissaTmp < FP8_MANTISSA_MASK)
            {
                exp = (int)(exp + FP8_EXPONENT_BIAS);
                u = (byte)((exp << FP8_NUM_MANTISSA_BITS) | mantissaTmp);
                may_be_nan = mantissaTmp == FP8_MANTISSA_MASK - 1;
            }
            else
            {
                return FromRaw((byte)(sign | FP8_MAX_FLT));
            }
        }
        else
        {
            return FromRaw((byte)(sign | FP8_MAX_FLT));
        }

        // Round to nearest even
        const int NUM_BITS_SHIFT = FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1);
        int round_bit = (mantissa >> NUM_BITS_SHIFT) & 1;
        sticky_bit |= (mantissa & ((1 << NUM_BITS_SHIFT) - 1)) != 0 ? 1 : 0;

        if ((round_bit != 0 && sticky_bit != 0) || (round_bit != 0 && (u & 1) != 0))
        {
            u++;
            if (may_be_nan)
            {
                skip_sign = true;
            }
        }

        if (u > FP8_MAX_FLT)
        {
            u = (byte)(sign | FP8_MAX_FLT);
        }

        if (!skip_sign)
        {
            u |= sign;
        }

        return FromRaw(u);
    }

    public static explicit operator Float8E4M3(Half input)
    {
        return (Float8E4M3)(float)input;
    }

    public static explicit operator Half(Float8E4M3 input)
    {
        return (Half)(float)input;
    }

    public static bool operator ==(Float8E4M3 lhs, Float8E4M3 rhs) => lhs._value == rhs._value;

    public static bool operator !=(Float8E4M3 lhs, Float8E4M3 rhs) => lhs._value != rhs._value;

    public static bool operator <(Float8E4M3 left, Float8E4M3 right)
    {
        return left.CompareTo(right) < 0;
    }

    public static bool operator <=(Float8E4M3 left, Float8E4M3 right)
    {
        return left.CompareTo(right) <= 0;
    }

    public static bool operator >(Float8E4M3 left, Float8E4M3 right)
    {
        return left.CompareTo(right) > 0;
    }

    public static bool operator >=(Float8E4M3 left, Float8E4M3 right)
    {
        return left.CompareTo(right) >= 0;
    }

    public static Float8E4M3 operator %(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    public static Float8E4M3 operator +(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    public static Float8E4M3 operator --(Float8E4M3 value) => throw new NotImplementedException();

    public static Float8E4M3 operator /(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    public static Float8E4M3 operator ++(Float8E4M3 value) => throw new NotImplementedException();

    public static Float8E4M3 operator *(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    public static Float8E4M3 operator -(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    public static Float8E4M3 operator -(Float8E4M3 value) => throw new NotImplementedException();

    public static Float8E4M3 operator +(Float8E4M3 value) => throw new NotImplementedException();

    static bool IComparisonOperators<Float8E4M3, Float8E4M3, bool>.operator >(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static bool IComparisonOperators<Float8E4M3, Float8E4M3, bool>.operator >=(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static bool IComparisonOperators<Float8E4M3, Float8E4M3, bool>.operator <(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static bool IComparisonOperators<Float8E4M3, Float8E4M3, bool>.operator <=(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static Float8E4M3 IModulusOperators<Float8E4M3, Float8E4M3, Float8E4M3>.operator %(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static Float8E4M3 IAdditionOperators<Float8E4M3, Float8E4M3, Float8E4M3>.operator +(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static Float8E4M3 IDecrementOperators<Float8E4M3>.operator --(Float8E4M3 value) => throw new NotImplementedException();

    static Float8E4M3 IDivisionOperators<Float8E4M3, Float8E4M3, Float8E4M3>.operator /(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static bool IEqualityOperators<Float8E4M3, Float8E4M3, bool>.operator ==(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static bool IEqualityOperators<Float8E4M3, Float8E4M3, bool>.operator !=(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static Float8E4M3 IIncrementOperators<Float8E4M3>.operator ++(Float8E4M3 value) => throw new NotImplementedException();

    static Float8E4M3 IMultiplyOperators<Float8E4M3, Float8E4M3, Float8E4M3>.operator *(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static Float8E4M3 ISubtractionOperators<Float8E4M3, Float8E4M3, Float8E4M3>.operator -(Float8E4M3 left, Float8E4M3 right) => throw new NotImplementedException();

    static Float8E4M3 IUnaryNegationOperators<Float8E4M3, Float8E4M3>.operator -(Float8E4M3 value) => throw new NotImplementedException();

    static Float8E4M3 IUnaryPlusOperators<Float8E4M3, Float8E4M3>.operator +(Float8E4M3 value) => throw new NotImplementedException();

    public static Float8E4M3 Abs(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsCanonical(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsComplexNumber(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsEvenInteger(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsFinite(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsImaginaryNumber(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsInfinity(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsInteger(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsNaN(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsNegative(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsNegativeInfinity(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsNormal(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsOddInteger(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsPositive(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsPositiveInfinity(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsRealNumber(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsSubnormal(Float8E4M3 value) => throw new NotImplementedException();

    public static bool IsZero(Float8E4M3 value) => throw new NotImplementedException();

    public static Float8E4M3 MaxMagnitude(Float8E4M3 x, Float8E4M3 y) => throw new NotImplementedException();

    public static Float8E4M3 MaxMagnitudeNumber(Float8E4M3 x, Float8E4M3 y) => throw new NotImplementedException();

    public static Float8E4M3 MinMagnitude(Float8E4M3 x, Float8E4M3 y) => throw new NotImplementedException();

    public static Float8E4M3 MinMagnitudeNumber(Float8E4M3 x, Float8E4M3 y) => throw new NotImplementedException();

    public static Float8E4M3 Parse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider) => throw new NotImplementedException();

    public static Float8E4M3 Parse(string s, NumberStyles style, IFormatProvider? provider) => throw new NotImplementedException();

    public static bool TryParse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider, [MaybeNullWhen(false)] out Float8E4M3 result) => throw new NotImplementedException();

    public static bool TryParse([NotNullWhen(true)] string? s, NumberStyles style, IFormatProvider? provider, [MaybeNullWhen(false)] out Float8E4M3 result) => throw new NotImplementedException();

    public static Float8E4M3 Parse(ReadOnlySpan<char> s, IFormatProvider? provider) => throw new NotImplementedException();

    public static bool TryParse(ReadOnlySpan<char> s, IFormatProvider? provider, [MaybeNullWhen(false)] out Float8E4M3 result) => throw new NotImplementedException();

    public static Float8E4M3 Parse(string s, IFormatProvider? provider) => throw new NotImplementedException();

    public static bool TryParse([NotNullWhen(true)] string? s, IFormatProvider? provider, [MaybeNullWhen(false)] out Float8E4M3 result) => throw new NotImplementedException();

    static Float8E4M3 INumberBase<Float8E4M3>.Abs(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsCanonical(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsComplexNumber(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsEvenInteger(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsFinite(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsImaginaryNumber(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsInfinity(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsInteger(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsNaN(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsNegative(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsNegativeInfinity(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsNormal(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsOddInteger(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsPositive(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsPositiveInfinity(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsRealNumber(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsSubnormal(Float8E4M3 value) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.IsZero(Float8E4M3 value) => throw new NotImplementedException();

    static Float8E4M3 INumberBase<Float8E4M3>.MaxMagnitude(Float8E4M3 x, Float8E4M3 y) => throw new NotImplementedException();

    static Float8E4M3 INumberBase<Float8E4M3>.MaxMagnitudeNumber(Float8E4M3 x, Float8E4M3 y) => throw new NotImplementedException();

    static Float8E4M3 INumberBase<Float8E4M3>.MinMagnitude(Float8E4M3 x, Float8E4M3 y) => throw new NotImplementedException();

    static Float8E4M3 INumberBase<Float8E4M3>.MinMagnitudeNumber(Float8E4M3 x, Float8E4M3 y) => throw new NotImplementedException();

    static Float8E4M3 INumberBase<Float8E4M3>.Parse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider) => throw new NotImplementedException();

    static Float8E4M3 INumberBase<Float8E4M3>.Parse(string s, NumberStyles style, IFormatProvider? provider) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.TryConvertFromChecked<TOther>(TOther value, out Float8E4M3 result) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.TryConvertFromSaturating<TOther>(TOther value, out Float8E4M3 result) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.TryConvertFromTruncating<TOther>(TOther value, out Float8E4M3 result) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.TryConvertToChecked<TOther>(Float8E4M3 value, out TOther result) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.TryConvertToSaturating<TOther>(Float8E4M3 value, out TOther result) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.TryConvertToTruncating<TOther>(Float8E4M3 value, out TOther result) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.TryParse(ReadOnlySpan<char> s, NumberStyles style, IFormatProvider? provider, out Float8E4M3 result) => throw new NotImplementedException();

    static bool INumberBase<Float8E4M3>.TryParse(string? s, NumberStyles style, IFormatProvider? provider, out Float8E4M3 result) => throw new NotImplementedException();

    static Float8E4M3 ISpanParsable<Float8E4M3>.Parse(ReadOnlySpan<char> s, IFormatProvider? provider) => throw new NotImplementedException();

    static bool ISpanParsable<Float8E4M3>.TryParse(ReadOnlySpan<char> s, IFormatProvider? provider, out Float8E4M3 result) => throw new NotImplementedException();

    static Float8E4M3 IParsable<Float8E4M3>.Parse(string s, IFormatProvider? provider) => throw new NotImplementedException();

    static bool IParsable<Float8E4M3>.TryParse(string? s, IFormatProvider? provider, out Float8E4M3 result) => throw new NotImplementedException();

    /// <summary>
    /// Reinterpret cast <see cref="byte"/> to <see cref="Float8E4M3"/>.
    /// </summary>
    /// <param name="value">Byte value.</param>
    /// <returns>Casted Float8E4M3.</returns>
    public static Float8E4M3 FromRaw(byte value)
    {
        Float8E4M3 result;
        Unsafe.SkipInit(out result);
        result._value = value;
        return result;
    }

    public byte ToRaw()
    {
        return _value;
    }

    public bool Equals(Float8E4M3 other)
    {
        return other == this;
    }

    public override bool Equals(object? obj)
    {
        if (obj is Float8E4M3)
        {
            var fp8 = (Float8E4M3)obj;
            return fp8 == this;
        }

        return false;
    }

    public override int GetHashCode()
    {
        return _value.GetHashCode();
    }

    public override string ToString()
    {
        return ((float)this).ToString();
    }

    public int CompareTo(Float8E4M3 other)
    {
        return ((float)this).CompareTo((float)other);
    }

    public int CompareTo(object? obj) => throw new NotImplementedException();

    public bool TryFormat(Span<char> destination, out int charsWritten, ReadOnlySpan<char> format, IFormatProvider? provider) => throw new NotImplementedException();

    public string ToString(string? format, IFormatProvider? formatProvider) => throw new NotImplementedException();

    int IComparable.CompareTo(object? obj) => throw new NotImplementedException();

    int IComparable<Float8E4M3>.CompareTo(Float8E4M3 other) => throw new NotImplementedException();

    bool IEquatable<Float8E4M3>.Equals(Float8E4M3 other) => this.Equals(other);

    bool ISpanFormattable.TryFormat(Span<char> destination, out int charsWritten, ReadOnlySpan<char> format, IFormatProvider? provider) => throw new NotImplementedException();

    string IFormattable.ToString(string? format, IFormatProvider? formatProvider) => throw new NotImplementedException();
}

/// <summary>
/// Float8E4M3.
/// </summary>
public struct Float8E5M2 : IEquatable<Float8E5M2>, IComparable<Float8E5M2>
{
    /// <summary>
    /// FP8 E4M3 representation bits.
    /// </summary>
    public byte _value;

    public static Float8E5M2 NaN => FromRaw(0b1111111);

    public static Float8E5M2 Infinity => NaN;

    public static Float8E5M2 NegInfinity => NaN;

    public static Float8E5M2 Zero => FromRaw(0b0000000);

    public static Float8E5M2 MaxNormal => FromRaw(0b1111110);

    public static Float8E5M2 MinNormal => FromRaw(0b0010000);

    public static Float8E5M2 MaxSubnormal => FromRaw(0b0000111);

    public static Float8E5M2 MinSubnormal => FromRaw(0b0000001);

    /// <summary>
    /// Implicit conversion from Float8E5M2 to float.
    /// </summary>
    /// <param name="input">FP8 value.</param>
    public static implicit operator float(Float8E5M2 input)
    {
        const bool IS_E4M3 = false;

        // Number of Bits representing mantissa and exponents
        const int FP32_NUM_BITS = 32;
        const int FP32_NUM_MANTISSA_BITS = 23;
        const int FP32_EXPONENT_BIAS = 127;

        const int FP8_NUM_BITS = 8;
        const int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;
        const int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;
        const int FP8_EXPONENT_BIAS = IS_E4M3 ? 7 : 15;
        const int FP8_MAX_EXPONENT = IS_E4M3 ? 7 : 15;
        const byte FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1;
        const byte FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1;

        const uint kF32_NaN = 0x7fffffff;

        byte f8 = input._value;
        int sign = (f8 >> (FP8_NUM_BITS - 1)) & 1;
        int exp = (f8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK;
        int mantissa = f8 & FP8_MANTISSA_MASK;
        uint f = (uint)(sign << (FP32_NUM_BITS - 1));

        if (IS_E4M3 && exp == 15 && mantissa == 0x7)
        {
            // Handle special case for NaN in E4M3 format
            f = kF32_NaN;
        }
        else if (exp > 0 && (IS_E4M3 || exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1)))
        {
            // Normal case
            exp += FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS;
            f = (uint)(f | ((uint)exp << FP32_NUM_MANTISSA_BITS) | ((uint)mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
        }
        else if (exp == 0)
        {
            if (mantissa != 0)
            {
                // Subnormal case
                exp += FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS + 1;
                while ((mantissa & (1 << FP8_NUM_MANTISSA_BITS)) == 0)
                {
                    mantissa <<= 1;
                    exp--;
                }

                mantissa &= FP8_MANTISSA_MASK;
                f = (uint)(f | ((uint)exp << FP32_NUM_MANTISSA_BITS) | ((uint)mantissa << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
            }

            // Sign-preserving zero is already handled by f being zeroed
        }
        else
        {
            if (mantissa == 0)
            {
                // Sign-preserving infinity
                f |= 0x7f800000;
            }
            else
            {
                // Canonical NaN
                f = kF32_NaN;
            }
        }

        // Return as float
        return BitConverter.ToSingle(BitConverter.GetBytes(f), 0);
    }

    /// <summary>
    /// Explicit conversion from float to Float8E5M2.
    /// </summary>
    /// <param name="input">Float value.</param>
    public static explicit operator Float8E5M2(float input)
    {
        const bool IS_E4M3 = false;

        // Number of Bits representing mantissa and exponents
        const int FP32_NUM_BITS = 32;
        const int FP32_NUM_MANTISSA_BITS = 23;
        const int FP32_EXPONENT_BIAS = 127;

        const int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;
        const int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;
        const int FP8_MAX_EXPONENT = IS_E4M3 ? 7 : 15;
        const int FP8_MIN_EXPONENT = IS_E4M3 ? -6 : -14;
        const int FP8_EXPONENT_BIAS = IS_E4M3 ? 7 : 15;

        const byte FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1;
        const byte FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1;

        const byte FP8_MAX_FLT = IS_E4M3 ? 0x7e : 0x7b;

        // Extract float bits using BitConverter
        uint s = BitConverter.ToUInt32(BitConverter.GetBytes(input), 0);

        // Extract sign, exponent and mantissa
        byte sign = (byte)((s >> 24) & 0x80);
        int exp = (int)(((s >> FP32_NUM_MANTISSA_BITS) & 0xff) - FP32_EXPONENT_BIAS);
        int mantissa = (int)(s & 0x7fffff);
        const byte kF8_NaN = 0x7f;

        // NaN => NaN
        if (float.IsNaN(input))
        {
            return FromRaw(kF8_NaN);
        }

        // Inf => MAX_FLT (satfinite)
        if (float.IsInfinity(input))
        {
            return FromRaw((byte)(sign | FP8_MAX_FLT));
        }

        // Special handling for exponent -128
        if (exp == -128)
        {
            return FromRaw((byte)(sign | FP8_MAX_FLT));
        }

        int sticky_bit = 0;
        bool skip_sign = false;
        bool may_be_nan = false;

        byte u;
        if (exp >= FP8_MIN_EXPONENT && exp <= FP8_MAX_EXPONENT)
        {
            // Normal fp32 to normal fp8
            exp = (int)(exp + FP8_EXPONENT_BIAS);
            u = (byte)((exp & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS);
            u = (byte)(u | (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
        }
        else if (exp < FP8_MIN_EXPONENT)
        {
            // fp32 to subnormal fp8
            int rshift = FP8_MIN_EXPONENT - exp;
            if (rshift < FP32_NUM_BITS)
            {
                mantissa |= 1 << FP32_NUM_MANTISSA_BITS;
                sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0) ? 1 : 0;
                mantissa = mantissa >> rshift;
                u = (byte)((mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK);
            }
            else
            {
                mantissa = 0;
                u = 0;
            }
        }
        else if (exp == FP8_MAX_EXPONENT + 1)
        {
            byte mantissaTmp = (byte)(mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
            if (mantissaTmp < FP8_MANTISSA_MASK)
            {
                exp = (int)(exp + FP8_EXPONENT_BIAS);
                u = (byte)((exp << FP8_NUM_MANTISSA_BITS) | mantissaTmp);
                may_be_nan = mantissaTmp == FP8_MANTISSA_MASK - 1;
            }
            else
            {
                return FromRaw((byte)(sign | FP8_MAX_FLT));
            }
        }
        else
        {
            return FromRaw((byte)(sign | FP8_MAX_FLT));
        }

        // Round to nearest even
        const int NUM_BITS_SHIFT = FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1);
        int round_bit = (mantissa >> NUM_BITS_SHIFT) & 1;
        sticky_bit |= (mantissa & ((1 << NUM_BITS_SHIFT) - 1)) != 0 ? 1 : 0;

        if ((round_bit != 0 && sticky_bit != 0) || (round_bit != 0 && (u & 1) != 0))
        {
            u++;
            if (may_be_nan)
            {
                skip_sign = true;
            }
        }

        if (u > FP8_MAX_FLT)
        {
            u = (byte)(sign | FP8_MAX_FLT);
        }

        if (!skip_sign)
        {
            u |= sign;
        }

        return FromRaw(u);
    }

    public static explicit operator Float8E5M2(Half input)
    {
        return (Float8E5M2)(float)input;
    }

    public static explicit operator Half(Float8E5M2 input)
    {
        return (Half)(float)input;
    }

    public static bool operator ==(Float8E5M2 lhs, Float8E5M2 rhs) => lhs._value == rhs._value;

    public static bool operator !=(Float8E5M2 lhs, Float8E5M2 rhs) => lhs._value != rhs._value;

    public static bool operator <(Float8E5M2 left, Float8E5M2 right)
    {
        return left.CompareTo(right) < 0;
    }

    public static bool operator <=(Float8E5M2 left, Float8E5M2 right)
    {
        return left.CompareTo(right) <= 0;
    }

    public static bool operator >(Float8E5M2 left, Float8E5M2 right)
    {
        return left.CompareTo(right) > 0;
    }

    public static bool operator >=(Float8E5M2 left, Float8E5M2 right)
    {
        return left.CompareTo(right) >= 0;
    }

    /// <summary>
    /// Reinterpret cast <see cref="byte"/> to <see cref="Float8E5M2"/>.
    /// </summary>
    /// <param name="value">Byte value.</param>
    /// <returns>Casted Float8E5M2.</returns>
    public static Float8E5M2 FromRaw(byte value)
    {
        Float8E5M2 result;
        Unsafe.SkipInit(out result);
        result._value = value;
        return result;
    }

    public byte ToRaw()
    {
        return _value;
    }

    public bool Equals(Float8E5M2 other)
    {
        return other == this;
    }

    public override bool Equals(object? obj)
    {
        if (obj is Float8E5M2)
        {
            var fp8 = (Float8E5M2)obj;
            return fp8 == this;
        }

        return false;
    }

    public override int GetHashCode()
    {
        return _value.GetHashCode();
    }

    public override string ToString()
    {
        return ((float)this).ToString();
    }

    public int CompareTo(Float8E5M2 other)
    {
        return ((float)this).CompareTo((float)other);
    }
}
