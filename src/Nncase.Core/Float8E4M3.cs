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
/// Float8E4M3.
/// </summary>
public struct Float8E4M3 : IEquatable<Float8E4M3>, IComparable<Float8E4M3>
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
}
