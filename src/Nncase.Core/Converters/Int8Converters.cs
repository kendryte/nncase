// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Converters;

internal class Int8Converters :
    ISpanConverter<sbyte, bool>,
    ISpanConverter<sbyte, sbyte>,
    ISpanConverter<sbyte, byte>,
    ISpanConverter<sbyte, short>,
    ISpanConverter<sbyte, ushort>,
    ISpanConverter<sbyte, int>,
    ISpanConverter<sbyte, uint>,
    ISpanConverter<sbyte, long>,
    ISpanConverter<sbyte, ulong>,
    ISpanConverter<sbyte, Half>,
    ISpanConverter<sbyte, float>,
    ISpanConverter<sbyte, double>,
    ISpanConverter<sbyte, BFloat16>
{
    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<bool> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<sbyte> dest, CastMode castMode)
    {
        source.CopyTo(dest);
    }

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<byte> dest, CastMode castMode)
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
                dest[i] = checked((byte)source[i]);
            }
        }
        else
        {
            for (int i = 0; i < source.Length; i++)
            {
                dest[i] = (byte)source[i];
            }
        }
    }

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<short> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<ushort> dest, CastMode castMode)
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
                dest[i] = checked((ushort)source[i]);
            }
        }
        else
        {
            for (int i = 0; i < source.Length; i++)
            {
                dest[i] = (ushort)source[i];
            }
        }
    }

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<int> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<uint> dest, CastMode castMode)
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
                dest[i] = checked((uint)source[i]);
            }
        }
        else
        {
            for (int i = 0; i < source.Length; i++)
            {
                dest[i] = (uint)source[i];
            }
        }
    }

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<long> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<ulong> dest, CastMode castMode)
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
                dest[i] = checked((ulong)source[i]);
            }
        }
        else
        {
            for (int i = 0; i < source.Length; i++)
            {
                dest[i] = (ulong)source[i];
            }
        }
    }

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<Half> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<float> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<double> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<sbyte> source, Span<BFloat16> dest, CastMode castMode)
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
