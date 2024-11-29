// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace Nncase.Utilities;

public static class SpanUtility
{
    public static ReadOnlySpan<TTo> UnsafeCast<TFrom, TTo>(ReadOnlySpan<TFrom> froms)
        where TFrom : class
        where TTo : class
    {
        ref var first = ref MemoryMarshal.GetReference(froms);
        ref var castFirst = ref Unsafe.As<TFrom, TTo>(ref first);
        return MemoryMarshal.CreateReadOnlySpan(ref castFirst, froms.Length);
    }

    public static void Deserialize<T>(Span<T> span, Stream stream)
        where T : unmanaged
    {
        var position = 0;
        while (position < span.Length)
        {
            var length = Math.Min(span.Length - position, 1024 * 1024 * 16);
            stream.ReadExactly(span.Slice(position, length).AsBytes());
            position += length;
        }
    }

    public static void Serialize<T>(ReadOnlySpan<T> span, Stream stream)
        where T : unmanaged
    {
        var position = 0;
        while (position < span.Length)
        {
            var length = Math.Min(span.Length - position, 1024 * 1024 * 16);
            stream.Write(span.Slice(position, length).AsBytes());
            position += length;
        }
    }
}
