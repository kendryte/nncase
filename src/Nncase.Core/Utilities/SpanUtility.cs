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

#pragma warning disable CS8500
    public static unsafe void Deserialize<T>(Span<T> span, Stream stream)
        where T : struct
    {
        if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
        {
            throw new NotSupportedException("Span<T> with reference type is not supported.");
        }

        var position = 0;
        while (position < span.Length)
        {
            var length = Math.Min(span.Length - position, 1024 * 1024 * 1024 / Unsafe.SizeOf<T>());
            fixed (T* ptr = &MemoryMarshal.GetReference(span.Slice(position)))
            {
                stream.ReadExactly(new Span<byte>(ptr, length * Unsafe.SizeOf<T>()));
                position += length;
            }
        }
    }

    public static unsafe void Serialize<T>(ReadOnlySpan<T> span, Stream stream)
        where T : struct
    {
        if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
        {
            throw new NotSupportedException("Span<T> with reference type is not supported.");
        }

        var position = 0;
        while (position < span.Length)
        {
            var length = Math.Min(span.Length - position, 1024 * 1024 * 1024 / Unsafe.SizeOf<T>());
            fixed (T* ptr = &MemoryMarshal.GetReference(span.Slice(position)))
            {
                stream.Write(new Span<byte>(ptr, length * Unsafe.SizeOf<T>()));
                position += length;
            }
        }
    }
#pragma warning restore CS8500 // 这会获取托管类型的地址、获取其大小或声明指向它的指针
}
