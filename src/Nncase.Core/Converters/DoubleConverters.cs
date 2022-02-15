// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Converters;

internal class DoubleConverters :
    ISpanConverter<double, bool>,
    ISpanConverter<double, sbyte>,
    ISpanConverter<double, byte>,
    ISpanConverter<double, short>,
    ISpanConverter<double, ushort>,
    ISpanConverter<double, int>,
    ISpanConverter<double, uint>,
    ISpanConverter<double, long>,
    ISpanConverter<double, ulong>,
    ISpanConverter<double, Half>,
    ISpanConverter<double, float>,
    ISpanConverter<double, double>,
    ISpanConverter<double, BFloat16>
{
    public void ConvertTo(ReadOnlySpan<double> source, Span<bool> dest, CastMode castMode)
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
            dest[i] = source[i] != 0.0;
        }
    }

    public void ConvertTo(ReadOnlySpan<double> source, Span<sbyte> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<double> source, Span<byte> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<double> source, Span<short> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<double> source, Span<ushort> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<double> source, Span<int> dest, CastMode castMode)
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
                dest[i] = checked((int)source[i]);
            }
        }
        else
        {
            for (int i = 0; i < source.Length; i++)
            {
                dest[i] = (int)source[i];
            }
        }
    }

    public void ConvertTo(ReadOnlySpan<double> source, Span<uint> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<double> source, Span<long> dest, CastMode castMode)
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
                dest[i] = checked((long)source[i]);
            }
        }
        else
        {
            for (int i = 0; i < source.Length; i++)
            {
                dest[i] = (long)source[i];
            }
        }
    }

    public void ConvertTo(ReadOnlySpan<double> source, Span<ulong> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<double> source, Span<Half> dest, CastMode castMode)
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
            dest[i] = (Half)source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<double> source, Span<float> dest, CastMode castMode)
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
            dest[i] = (float)source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<double> source, Span<double> dest, CastMode castMode)
    {
        source.CopyTo(dest);
    }

    public void ConvertTo(ReadOnlySpan<double> source, Span<BFloat16> dest, CastMode castMode)
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
            dest[i] = (BFloat16)source[i];
        }
    }
}
