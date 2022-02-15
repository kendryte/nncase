// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Converters;

internal class UInt8Converters :
    ISpanConverter<byte, bool>,
    ISpanConverter<byte, sbyte>,
    ISpanConverter<byte, byte>,
    ISpanConverter<byte, short>,
    ISpanConverter<byte, ushort>,
    ISpanConverter<byte, int>,
    ISpanConverter<byte, uint>,
    ISpanConverter<byte, long>,
    ISpanConverter<byte, ulong>,
    ISpanConverter<byte, Half>,
    ISpanConverter<byte, float>,
    ISpanConverter<byte, double>,
    ISpanConverter<byte, BFloat16>
{
    public void ConvertTo(ReadOnlySpan<byte> source, Span<bool> dest, CastMode castMode)
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
            dest[i] = source[i] != 0;
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<sbyte> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        if (castMode == CastMode.CheckOverflow)
        {
            for (int i = 0; i < source.Length; i++)
            {
                dest[i] = checked((sbyte)source[i]);
            }
        }
        else
        {
            for (int i = 0; i < source.Length; i++)
            {
                dest[i] = (sbyte)source[i];
            }
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<byte> dest, CastMode castMode)
    {
        source.CopyTo(dest);
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<short> dest, CastMode castMode)
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
            dest[i] = source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<ushort> dest, CastMode castMode)
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
            dest[i] = source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<int> dest, CastMode castMode)
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
            dest[i] = source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<uint> dest, CastMode castMode)
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
            dest[i] = source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<long> dest, CastMode castMode)
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
            dest[i] = source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<ulong> dest, CastMode castMode)
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
            dest[i] = source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<Half> dest, CastMode castMode)
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
            dest[i] = (Half)(float)source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<float> dest, CastMode castMode)
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
            dest[i] = source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<double> dest, CastMode castMode)
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
            dest[i] = source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<byte> source, Span<BFloat16> dest, CastMode castMode)
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
            dest[i] = (BFloat16)(float)source[i];
        }
    }
}
