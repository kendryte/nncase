using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Converter
{
    public ref struct SpanReader
    {
        private ReadOnlySpan<byte> _span;

        public bool IsCosumed => _span.IsEmpty;

        public int AvalilableBytes => _span.Length;

        public SpanReader(ReadOnlySpan<byte> span)
        {
            _span = span;
        }

        public byte PeekAsUInt8() =>
            Peek<byte>();

        public byte ReadAsUInt8() =>
            Read<byte>();

        public ushort ReadAsUInt16() =>
            Read<ushort>().ToBigEndian();

        public uint ReadAsUInt32() =>
            Read<uint>().ToBigEndian();

        public ulong ReadAsUInt64() =>
            Read<ulong>().ToBigEndian();

        public byte[] ReadAsBytes(int length) =>
            ReadAsSpan(length).ToArray();

        public SpanReader ReadAsSubReader(int length) =>
            new SpanReader(ReadAsSpan(length));

        public IPAddress ReadAsIPAddress() =>
            new IPAddress(Read<uint>());

        public string ReadAsString()
        {
            var len = ReadAsUInt16();
            return Encoding.UTF8.GetString(ReadAsBytes(len));
        }

        public byte[] ReadAsKey()
        {
            var len = ReadAsUInt16();
            return ReadAsBytes(len);
        }

        public ReadOnlySpan<byte> ReadAsSpan(int length)
        {
            var span = _span.Slice(0, length);
            Advance(length);
            return span;
        }

        public ReadOnlySpan<byte> ReadAsSpan() =>
            ReadAsSpan(AvalilableBytes);

        public void Skip(int count) =>
            Advance(count);

        private unsafe T Peek<T>() where T : unmanaged
        {
            var length = sizeof(T);
            var bytes = _span.Slice(0, length);
            var value = MemoryMarshal.Read<T>(bytes);
            return value;
        }

        public unsafe T Read<T>() where T : unmanaged
        {
            var length = sizeof(T);
            var bytes = _span.Slice(0, length);
            var value = MemoryMarshal.Read<T>(bytes);
            Advance(length);
            return value;
        }

        private void Advance(int count)
        {
            _span = _span.Slice(count);
        }
    }
}
