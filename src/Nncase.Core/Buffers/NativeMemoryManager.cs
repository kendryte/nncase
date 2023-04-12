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

internal unsafe class NativeMemoryManager<T> : MemoryManager<T>
    where T : unmanaged
{
    private readonly int _length;
    private IntPtr _pointer;

    public NativeMemoryManager(int length)
    {
        if (length < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }

        if (length != 0)
        {
            _pointer = Marshal.AllocHGlobal((nint)sizeof(T) * length);
            _length = length;
            GC.AddMemoryPressure((long)sizeof(T) * length);
        }
    }

    /// <summary>
    /// Finalizes an instance of the <see cref="NativeMemoryManager{T}"/> class.
    /// </summary>
#pragma warning disable CA2015 // Used only in DenseTensor
    ~NativeMemoryManager()
#pragma warning restore CA2015
    {
        Dispose(false);
    }

    public IntPtr Pointer => _pointer;

    public override Span<T> GetSpan()
    {
        if (_length == 0)
        {
            return Span<T>.Empty;
        }

        return new Span<T>((void*)_pointer, _length);
    }

    public override MemoryHandle Pin(int elementIndex = 0)
    {
        if ((uint)elementIndex > (uint)_length)
        {
            throw new ArgumentOutOfRangeException(nameof(elementIndex));
        }

        return new MemoryHandle(Unsafe.Add<T>((void*)_pointer, elementIndex), default, this);
    }

    public override void Unpin()
    {
    }

    protected override void Dispose(bool disposing)
    {
        var pointer = Interlocked.Exchange(ref _pointer, IntPtr.Zero);
        if (pointer != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(pointer);
            GC.RemoveMemoryPressure((long)sizeof(T) * _length);
        }
    }
}
