// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Converters;

internal class Float8Converters :
    ISpanConverter<Float8E4M3, Half>,
    ISpanConverter<Float8E4M3, float>,
    ISpanConverter<Float8E4M3, Float8E4M3>
{
    public void ConvertTo(ReadOnlySpan<Float8E4M3> source, Span<Half> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<Float8E4M3> source, Span<float> dest, CastMode castMode)
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

    public void ConvertTo(ReadOnlySpan<Float8E4M3> source, Span<Float8E4M3> dest, CastMode castMode)
    {
        source.CopyTo(dest);
    }
}
