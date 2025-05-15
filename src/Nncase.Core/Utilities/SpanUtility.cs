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
#pragma warning restore CS8500

    public static T[] Concat<T>(ReadOnlySpan<T> first, ReadOnlySpan<T> second)
    {
        var result = new T[first.Length + second.Length];
        first.CopyTo(result);
        second.CopyTo(result.AsSpan(first.Length));
        return result;
    }

    public static ReadOnlySpan<T> AsReadOnlySpan<T>(this T[] array)
    {
        return MemoryMarshal.CreateReadOnlySpan(ref MemoryMarshal.GetArrayDataReference(array), array.Length);
    }

    public static ReadOnlySpan<T> AsReadOnlySpan<T>(this T[] array, int start, int length)
    {
        if (start < 0 || length < 0 || start + length > array.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(start), "Start or length is out of range.");
        }

        return MemoryMarshal.CreateReadOnlySpan(ref MemoryMarshal.GetArrayDataReference(array), array.Length).Slice(start, length);
    }

    public static bool ReferenceContains<T>(this ReadOnlySpan<T> span, T value)
        where T : class
    {
        foreach (var item in span)
        {
            if (ReferenceEquals(item, value))
            {
                return true;
            }
        }

        return false;
    }
}
