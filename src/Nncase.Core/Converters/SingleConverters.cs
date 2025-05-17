// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Converters;

internal class SingleConverters :
    ISpanConverter<float, bool>,
    ISpanConverter<float, sbyte>,
    ISpanConverter<float, byte>,
    ISpanConverter<float, short>,
    ISpanConverter<float, ushort>,
    ISpanConverter<float, int>,
    ISpanConverter<float, uint>,
    ISpanConverter<float, long>,
    ISpanConverter<float, ulong>,
    ISpanConverter<float, Half>,
    ISpanConverter<float, float>,
    ISpanConverter<float, double>,
    ISpanConverter<float, BFloat16>,
    ISpanConverter<float, Float8E4M3>,
    ISpanConverter<float, Float8E5M2>,
    ISpanConverter<float, Vector32<float>>,
    ISpanConverter<Vector32<float>, float>,
    ISpanConverter<float, Vector64<float>>,
    ISpanConverter<Vector64<float>, float>
{
    public void ConvertTo(ReadOnlySpan<float> source, Span<bool> dest, CastMode castMode)
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
            dest[i] = source[i] != 0.0f;
        }
    }

    public void ConvertTo(ReadOnlySpan<float> source, Span<sbyte> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<byte> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<short> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<ushort> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<int> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<uint> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<long> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<ulong> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<Half> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<float> dest, CastMode castMode)
    {
        source.CopyTo(dest);
    }

    public void ConvertTo(ReadOnlySpan<float> source, Span<double> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<Float8E4M3> dest, CastMode castMode)
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
            dest[i] = (Float8E4M3)source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<float> source, Span<Float8E5M2> dest, CastMode castMode)
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
            dest[i] = (Float8E5M2)source[i];
        }
    }

    public void ConvertTo(ReadOnlySpan<float> source, Span<BFloat16> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<float> source, Span<Vector32<float>> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        var elementsPerVector = Vector32<float>.Count;
        var requiredSourceSize = dest.Length * elementsPerVector;

        if (source.Length < requiredSourceSize)
        {
            throw new ArgumentException("Source buffer does not contain enough elements to fill the vectors");
        }

        for (int i = 0; i < dest.Length; i++)
        {
            var vector = default(Vector32<float>);
            ConvertTo(source.Slice(i * elementsPerVector, elementsPerVector), vector.AsSpan(), castMode);
        }
    }

    public void ConvertTo(ReadOnlySpan<Vector32<float>> source, Span<float> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        var elementsPerVector = Vector32<float>.Count;
        var requiredDestSize = source.Length * elementsPerVector;

        if (dest.Length < requiredDestSize)
        {
            throw new ArgumentException("Destination buffer is not large enough for the flattened vector data");
        }

        for (int i = 0; i < source.Length; i++)
        {
            var vector = source[i];
            ConvertTo(vector.AsSpan(), dest.Slice(i * elementsPerVector, elementsPerVector), castMode);
        }
    }

    public void ConvertTo(ReadOnlySpan<float> source, Span<Vector64<float>> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        var elementsPerVector = Vector64<float>.Count;
        var requiredSourceSize = dest.Length * elementsPerVector;

        if (source.Length < requiredSourceSize)
        {
            throw new ArgumentException("Source buffer does not contain enough elements to fill the vectors");
        }

        for (int i = 0; i < dest.Length; i++)
        {
            var vector = default(Vector64<float>);
            ConvertTo(source.Slice(i * elementsPerVector, elementsPerVector), vector.AsSpan(), castMode);
        }
    }

    public void ConvertTo(ReadOnlySpan<Vector64<float>> source, Span<float> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException();
        }

        var elementsPerVector = Vector64<float>.Count;
        var requiredDestSize = source.Length * elementsPerVector;

        if (dest.Length < requiredDestSize)
        {
            throw new ArgumentException("Destination buffer is not large enough for the flattened vector data");
        }

        for (int i = 0; i < source.Length; i++)
        {
            var vector = source[i];
            ConvertTo(vector.AsSpan(), dest.Slice(i * elementsPerVector, elementsPerVector), castMode);
        }
    }
}
