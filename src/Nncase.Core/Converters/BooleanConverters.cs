// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Converters;

internal class BooleanConverters :
    ISpanConverter<bool, bool>,
    ISpanConverter<bool, sbyte>,
    ISpanConverter<bool, byte>,
    ISpanConverter<bool, short>,
    ISpanConverter<bool, ushort>,
    ISpanConverter<bool, int>,
    ISpanConverter<bool, uint>,
    ISpanConverter<bool, long>,
    ISpanConverter<bool, ulong>,
    ISpanConverter<bool, Half>,
    ISpanConverter<bool, float>,
    ISpanConverter<bool, double>,
    ISpanConverter<bool, BFloat16>
{
    public void ConvertTo(ReadOnlySpan<bool> source, Span<bool> dest, CastMode castMode)
    {
        source.CopyTo(dest);
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<sbyte> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? (sbyte)1 : (sbyte)0;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<byte> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? (byte)1 : (byte)0;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<short> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? (short)1 : (short)0;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<ushort> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? (ushort)1 : (ushort)0;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<int> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? 1 : 0;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<uint> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? 1U : 0;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<long> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? 1L : 0;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<ulong> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? 1UL : 0;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<Half> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? (Half)1f : (Half)0f;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<float> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? 1f : 0f;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<double> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? 1.0 : 0.0;
        }
    }

    public void ConvertTo(ReadOnlySpan<bool> source, Span<BFloat16> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        var one = (BFloat16)1f;
        var zero = (BFloat16)0f;

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = source[i] ? one : zero;
        }
    }
}
