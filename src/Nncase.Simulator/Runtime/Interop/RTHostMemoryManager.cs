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

namespace Nncase.Runtime.Interop;

internal unsafe class RTHostMemoryManager : MemoryManager<byte>
{
    private readonly uint _length;
    private RTHostBuffer? _buffer;
    private IntPtr _pointer;

    public RTHostMemoryManager(RTHostBuffer buffer, IntPtr pointer, uint length)
    {
        _buffer = buffer;
        _pointer = pointer;
        _length = length;

        if (length != 0)
        {
            GC.AddMemoryPressure(length);
        }
    }

    public IntPtr Pointer => _pointer;

    public override Span<byte> GetSpan()
    {
        if (_length == 0)
        {
            return Span<byte>.Empty;
        }

        return new Span<byte>((void*)_pointer, (int)_length);
    }

    public override MemoryHandle Pin(int elementIndex = 0)
    {
        if ((uint)elementIndex > _length)
        {
            throw new ArgumentOutOfRangeException(nameof(elementIndex));
        }

        return new MemoryHandle(Unsafe.Add<byte>((void*)_pointer, elementIndex), default, this);
    }

    public override void Unpin()
    {
    }

    protected override void Dispose(bool disposing)
    {
        var pointer = Interlocked.Exchange(ref _pointer, IntPtr.Zero);
        if (pointer != IntPtr.Zero && _buffer != null && _length != 0)
        {
            Native.HostBufferUnmap(_buffer.DangerousGetHandle());
            GC.RemoveMemoryPressure(_length);
            _buffer = null;
        }
    }
}
