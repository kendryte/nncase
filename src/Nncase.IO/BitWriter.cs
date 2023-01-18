// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace Nncase.IO
{
    /// <summary>
    /// the bit writer.
    /// </summary>
    public ref struct BitWriter
    {
        private Span<byte> _data;
        private ulong _buffer;
        private ulong _avail;

        /// <summary>
        /// Initializes a new instance of the <see cref="BitWriter"/> struct.
        /// ctor.
        /// </summary>
        /// <param name="data"> the ouput stream.</param>
        /// <param name="bitoffset">start bit offset.</param>
        public BitWriter(Span<byte> data, ulong bitoffset = 0)
        {
            _data = data;
            _buffer = 0;
            _avail = sizeof(ulong) * 8;
            if (bitoffset != 0)
            {
                _data = _data.Slice((int)(bitoffset / 8));
                bitoffset %= 8;
                _buffer = _data[0] & ((1UL << (int)bitoffset) - 1);
                _avail -= bitoffset;
            }
        }

        /// <summary>
        /// writhe the unmanaged value.
        /// </summary>
        public void Write<T>(T value, int bits)
            where T : unmanaged
        {
            WriteArray(MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref value, 1)), bits);
        }

        /// <summary>
        /// Flush the unwrited value.
        /// </summary>
        public void Flush()
        {
            var write_bytes = (Buffer_written_bits() + 7) / 8;
            if (write_bytes != 0)
            {
                Trace.Assert(_data.Length >= write_bytes);
                var bufferSpan = MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref _buffer, 1));
                for (int i = 0; i < write_bytes; i++)
                {
                    _data[i] = bufferSpan[i];
                }

                _data = _data.Slice(write_bytes);
                _buffer = 0;
                _avail = sizeof(ulong) * 8;
            }
        }

        private void WriteArray(ReadOnlySpan<byte> src, int bits)
        {
            while (bits > 0)
            {
                var to_write = Math.Min(bits, 8);
                Write_bits_le8(src[0], to_write);
                src = src.Slice(1);
                bits -= to_write;
            }
        }

        /// <summary>
        /// write the value less then 8.
        /// </summary>
        private void Write_bits_le8(byte value, int bits)
        {
            Trace.Assert(bits <= 8);
            Reserve_buffer_8();
            ulong new_value = value & ((1UL << bits) - 1);
            _buffer = _buffer | (new_value << Buffer_written_bits());
            _avail -= (ulong)bits;
        }

        /// <summary>
        /// create the new buffer for wirte.
        /// </summary>
        private void Reserve_buffer_8()
        {
            if (_avail < 8)
            {
                var write_bytes = Buffer_written_bits() / 8;
                Trace.Assert(_data.Length >= write_bytes);
                var bufferSpan = MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref _buffer, 1));
                for (int i = 0; i < write_bytes; i++)
                {
                    _data[i] = bufferSpan[i];
                }

                _data = _data.Slice(write_bytes);
                if (write_bytes == sizeof(ulong))
                {
                    _buffer = 0;
                }
                else
                {
                    _buffer >>= write_bytes * 8;
                }

                _avail += (ulong)write_bytes * 8;
            }
        }

        /// <summary>
        /// get current written bits.
        /// </summary>
        private int Buffer_written_bits()
        {
            return (int)((sizeof(ulong) * 8) - _avail);
        }
    }
}
