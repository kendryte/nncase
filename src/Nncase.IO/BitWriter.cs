using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace Nncase.IO
{
    /// <summary>
    /// the bit writer
    /// </summary>
    public ref struct BitWriter
    {
        Span<byte> _data;
        ulong _buffer;
        ulong _avail;

        /// <summary>
        /// ctor
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
                _buffer = _data[0] & (((ulong)1 << (int)bitoffset) - 1);
                _avail -= bitoffset;
            }
        }

        void WriteArray(ReadOnlySpan<byte> src, int bits)
        {
            while (bits > 0)
            {
                var to_write = Math.Min(bits, 8);
                write_bits_le8(src[0], to_write);
                src = src.Slice(1);
                bits -= to_write;
            }
        }

        /// <summary>
        /// writhe the unmanaged value.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="value"></param>
        /// <param name="bits"></param>
        public void Write<T>(T value, int bits) where T : unmanaged
        {
            WriteArray(MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref value, 1)), bits);
        }

        /// <summary>
        /// Flush the unwrited value.
        /// </summary>
        public void Flush()
        {
            var write_bytes = (buffer_written_bits() + 7) / 8;
            if (write_bytes != 0)
            {
                Debug.Assert(_data.Length >= write_bytes);
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

        /// <summary>
        /// write the value less then 8
        /// </summary>
        /// <param name="value"></param>
        /// <param name="bits"></param>
        void write_bits_le8(byte value, int bits)
        {
            Debug.Assert(bits <= 8);
            reserve_buffer_8();
            ulong new_value = value & (((ulong)(1) << bits) - 1);
            _buffer = _buffer | (new_value << buffer_written_bits());
            _avail -= (ulong)bits;
        }

        /// <summary>
        /// create the new buffer for wirte
        /// </summary>
        void reserve_buffer_8()
        {
            if (_avail < 8)
            {
                var write_bytes = buffer_written_bits() / 8;
                Debug.Assert(_data.Length >= write_bytes);
                var bufferSpan = MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref _buffer, 1));
                for (int i = 0; i < write_bytes; i++)
                {
                    _data[i] = bufferSpan[i];
                }
                _data = _data.Slice(write_bytes);
                if (write_bytes == sizeof(ulong))
                    _buffer = 0;
                else
                    _buffer >>= write_bytes * 8;
                _avail += (ulong)write_bytes * 8;
            }
        }

        /// <summary>
        /// get current written bits.
        /// </summary>
        /// <returns></returns>
        int buffer_written_bits()
        {
            return (int)((sizeof(ulong) * 8) - _avail);
        }
    }
}