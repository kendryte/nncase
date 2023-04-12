// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Buffers;

internal sealed class ArrayMemoryManager<T> : MemoryManager<T>
    where T : unmanaged
{
    private readonly Array _array;

    public ArrayMemoryManager(Array array)
    {
        _array = array;
    }

    public override Span<T> GetSpan()
    {
        ref var byteRef = ref MemoryMarshal.GetArrayDataReference(_array);
        ref var dataRef = ref Unsafe.As<byte, T>(ref byteRef);
        return MemoryMarshal.CreateSpan(ref dataRef, _array.Length);
    }

    public unsafe override MemoryHandle Pin(int elementIndex = 0)
    {
        var handle = GCHandle.Alloc(_array, GCHandleType.Pinned);
        ref var byteRef = ref MemoryMarshal.GetArrayDataReference(_array);
        ref var dataRef = ref Unsafe.Add(ref Unsafe.As<byte, T>(ref byteRef), elementIndex);
        return new MemoryHandle(Unsafe.AsPointer(ref dataRef), handle);
    }

    public override void Unpin()
    {
    }

    protected override void Dispose(bool disposing)
    {
    }
}
