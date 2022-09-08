// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using Nncase.Transform;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Quantization;

/// <summary>
/// quantization utility
/// </summary>
public static class Utility
{
    public static float GetCosineSimilarity(Span<float> V1, Span<float> V2)
    {
        int N = 0;
        N = ((V2.Length < V1.Length) ? V2.Length : V1.Length);
        float dot = 0.0f;
        float mag1 = 0.0f;
        float mag2 = 0.0f;
        for (int n = 0; n < N; n++)
        {
            dot += V1[n] * V2[n];
            mag1 += (float)Math.Pow(V1[n], 2);
            mag2 += (float)Math.Pow(V2[n], 2);
        }

        return dot / (float)(Math.Sqrt(mag1) * Math.Sqrt(mag2));
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="x"></param>
    /// <param name="e"></param>
    /// <returns></returns>
    private static double frexp(double x, ref int e)
    {
        // union { double d; uint64_t i; }
        ulong y = BitConverter.DoubleToUInt64Bits(x);
        // y = { x };
        int ee = (int)y >> 52 & 0x7ff;

        if (ee == 0)
        {
            if (x != 0)
            {
                x = frexp(x * 1.844674e+19, ref e);
                e -= 64;
            }
            else e = 0;
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

    /// <summary>
    /// get fixed mul struct instance.
    /// </summary>
    /// <param name="value"></param>
    /// <param name="max_bits"></param>
    /// <param name="max_shift"></param>
    /// <param name="is_signed"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static FixedMul GetFixedMul(float value, int max_bits, byte max_shift, bool is_signed)
    {
        if (!(is_signed || value >= 0))
            throw new InvalidOperationException();

        var bits = is_signed ? max_bits - 1 : max_bits;
        int shift = 0;
        float mul = 0;

        if (value == 0)
        {
            mul = 0;
            shift = 0;
        }
        else if (Math.Abs(value) > 1)
        {
            int mul_shift = 0;
            mul = checked((float)frexp((double)value, ref mul_shift));
            shift = Math.Min((int)max_shift, bits - mul_shift);
            mul = mul * (float)Math.Pow(2.0f, (float)(shift + mul_shift));
        }
        else
        {
            int mul_shift = 0;
            mul = checked((float)frexp(value, ref mul_shift));
            shift = Math.Min(max_shift + mul_shift, bits);
            mul = mul * (float)Math.Pow(2.0f, (float)shift);
            shift -= mul_shift;
        }

        if (!(Math.Abs(mul) < Math.Pow(2.0f, (float)bits)))
            throw new ArgumentOutOfRangeException();
        if (!(shift >= 0 && shift <= max_shift))
            throw new ArgumentOutOfRangeException();

        if (!(Math.Abs(value - mul * Math.Pow(2.0f, (float)-shift)) <= Math.E))
            throw new ArgumentOutOfRangeException();
        return new(mul, checked((sbyte)shift));
    }
}