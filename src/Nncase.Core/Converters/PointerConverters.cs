// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Converters;

internal class PointerConverters : IPointerSpanConverter<ulong>
{
    public void ConvertTo<T>(ReadOnlySpan<Pointer<T>> source, Span<ulong> dest, CastMode castMode)
        where T : unmanaged, IEquatable<T>
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
            dest[i] = source[i].Value;
        }
    }
}

internal class PointerIntConverters : IPointerSpanConverter<int>
{
    public void ConvertTo<T>(ReadOnlySpan<Pointer<T>> source, Span<int> dest, CastMode castMode)
        where T : unmanaged, IEquatable<T>
    {
        if (castMode != CastMode.KDefault)
        {
            throw new InvalidCastException();
        }

        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Dest buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            dest[i] = checked((int)source[i].Value);
        }
    }
}
