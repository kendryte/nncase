// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Nncase.IO;

public ref struct BitReader
{
    private ReadOnlySpan<byte> _data;
    private ulong _buffer;

    private ulong _avail;

    public BitReader(ReadOnlySpan<byte> data)
    {
        _data = data;
        _buffer = 0;
        _avail = 0;
    }

    /// <summary>
    /// Gets remain data length.
    /// </summary>
    public uint RemainBits => ((uint)_data.Length * 8) + (uint)_avail;

    /// <summary>
    /// read T from the span.
    /// </summary>
    public T Read<T>(ulong bits)
        where T : unmanaged
    {
        T ret = default;
        Read(MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan<T>(ref ret, 1)), bits);
        return ret;
    }

    private void Read(Span<byte> dest, ulong bits)
    {
        while (bits != 0)
        {
            var to_read = Math.Min(bits, 8UL);
            dest[0] = Read_bits_le8(to_read);
            dest = dest.Slice(1);
            bits -= to_read;
        }
    }

    private byte Read_bits_le8(ulong bits)
    {
        Trace.Assert(bits <= 8);

        Fill_buffer_le8(bits);
        byte ret = (byte)(_buffer & ((1UL << (int)bits) - 1));
        _buffer >>= (int)bits;
        _avail -= bits;
        return ret;
    }

    private void Fill_buffer_le8(ulong bits)
    {
        if (_avail < bits)
        {
            var max_read_bytes = Math.Min((ulong)_data.Length * 8, (sizeof(ulong) * 8) - _avail) / 8;
            Trace.Assert(max_read_bytes != 0);

            ulong tmp = 0;
            Memcpy(ref tmp, _data, (int)max_read_bytes);
            _data = _data.Slice((int)max_read_bytes);
            _buffer = _buffer | (tmp << (int)_avail);
            _avail += max_read_bytes * 8;
        }
    }

    private void Memcpy(ref ulong value, ReadOnlySpan<byte> data, int bits)
    {
        var valueSpan = MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan(ref value, 1));
        for (int i = 0; i < bits; i++)
        {
            valueSpan[i] = data[i];
        }
    }
}
