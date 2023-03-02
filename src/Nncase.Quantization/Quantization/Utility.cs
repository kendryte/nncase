// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Quantization;

/// <summary>
/// quantization utility.
/// </summary>
public static class Utility
{
    public static float GetCosineSimilarity(Span<float> v1, Span<float> v2)
    {
        int n1 = v2.Length < v1.Length ? v2.Length : v1.Length;
        double dot = 0.0f;
        double mag1 = 0.0f;
        double mag2 = 0.0f;
        for (int n = 0; n < n1; n++)
        {
            dot += (double)v1[n] * (double)v2[n];
            mag1 += Math.Pow((double)v1[n], 2);
            mag2 += Math.Pow((double)v2[n], 2);
        }

        if (dot == 0 && mag1 == 0 && mag2 == 0)
        {
            return 1.0f;
        }

        return (float)(dot / (Math.Sqrt(mag1) * Math.Sqrt(mag2)));
    }

    /// <summary>
    /// get fixed mul struct instance.
    /// </summary>
    public static FixedMul GetFixedMul(float value, int max_bits, byte max_shift, bool is_signed)
    {
        if (!(is_signed || value >= 0))
        {
            throw new InvalidOperationException();
        }

        var bits = is_signed ? max_bits - 1 : max_bits;
        int shift;
        float mul;
        if (value == 0)
        {
            mul = 0;
            shift = 0;
        }
        else if (Math.Abs(value) > 1)
        {
            int mul_shift = 0;
            mul = checked((float)Frexp((double)value, ref mul_shift));
            shift = Math.Min((int)max_shift, bits - mul_shift);
            mul = mul * (float)Math.Pow(2.0f, (float)(shift + mul_shift));
        }
        else
        {
            int mul_shift = 0;
            mul = checked((float)Frexp(value, ref mul_shift));
            shift = Math.Min(max_shift + mul_shift, bits);
            mul = mul * (float)Math.Pow(2.0f, (float)shift);
            shift -= mul_shift;
        }

        if (!(Math.Abs(mul) < Math.Pow(2.0f, (float)bits)))
        {
            throw new ArgumentOutOfRangeException(nameof(value));
        }

        if (!(shift >= 0 && shift <= max_shift))
        {
            throw new ArgumentOutOfRangeException(nameof(value));
        }

        if (!(Math.Abs(value - (mul * Math.Pow(2.0f, (float)-shift))) <= Math.E))
        {
            throw new ArgumentOutOfRangeException(nameof(value));
        }

        return new(mul, checked((sbyte)shift));
    }

    private static double Frexp(double x, ref int e)
    {
        // union { double d; uint64_t i; }
        ulong y = BitConverter.DoubleToUInt64Bits(x);

        // y = { x };
        int ee = ((int)y >> 52) & 0x7ff;

        if (ee == 0)
        {
            if (x != 0)
            {
                x = Frexp(x * 1.844674e+19, ref e);
                e -= 64;
            }
            else
            {
                e = 0;
            }

            return x;
        }
        else if (ee == 0x7ff)
        {
            return x;
        }

        e = ee - 0x3fe;
        y &= 0x800ffffffffffffful;
        y |= 0x3fe0000000000000ul;
        return BitConverter.UInt64BitsToDouble(y);
    }
}
