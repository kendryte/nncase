using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Nncase.IO
{

    public ref struct BitReader
    {

        public BitReader(ReadOnlySpan<byte> data)
        {
            _data = data;
            _buffer = 0;
            _avail = 0;
        }

        void Read(Span<byte> dest, ulong bits)
        {
            while (bits != 0)
            {
                var to_read = Math.Min(bits, (ulong)8);
                dest[0] = read_bits_le8(to_read);
                dest = dest.Slice(1);
                bits -= to_read;
            }
        }

        /// <summary>
        /// read T from the span.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="bits"></param>
        /// <returns></returns>
        public T Read<T>(ulong bits) where T : unmanaged
        {
            T ret = default;
            Read(MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan<T>(ref ret, 1)), bits);
            return ret;
        }

        byte read_bits_le8(ulong bits)
        {
            Debug.Assert(bits <= 8);

            fill_buffer_le8(bits);
            byte ret = (byte)(_buffer & (((ulong)1 << (int)bits) - 1));
            _buffer >>= (int)bits;
            _avail -= bits;
            return ret;
        }

        void fill_buffer_le8(ulong bits)
        {
            if (_avail < bits)
            {
                var max_read_bytes = Math.Min((ulong)_data.Length * 8, sizeof(ulong) * 8 - _avail) / 8;
                Debug.Assert(max_read_bytes != 0);

                ulong tmp = 0;
                memcpy(ref tmp, _data, (int)max_read_bytes);
                _data = _data.Slice((int)max_read_bytes);
                _buffer = _buffer | (tmp << (int)_avail);
                _avail += max_read_bytes * 8;
            }
        }

        void memcpy(ref ulong value, ReadOnlySpan<byte> data, int bits)
        {
            var valueSpan = MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan(ref value, 1));
            for (int i = 0; i < bits; i++)
            {
                valueSpan[i] = data[i];
            }
        }

        ReadOnlySpan<byte> _data;
        ulong _buffer;
        ulong _avail;
    }
}