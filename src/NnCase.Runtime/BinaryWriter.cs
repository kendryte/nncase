using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Runtime
{
    public class BinaryWriter : IDisposable
    {
        private readonly System.IO.BinaryWriter _writer;

        public long Position
        {
            get => _writer.BaseStream.Position;
            set => _writer.BaseStream.Position = value;
        }

        public BinaryWriter(Stream stream, bool leaveOpen = false)
        {
            _writer = new System.IO.BinaryWriter(stream, Encoding.UTF8, leaveOpen);
        }

        public void Write<T>(T value)
            where T : unmanaged
        {
            var span = MemoryMarshal.CreateReadOnlySpan(ref value, 1);
            _writer.Write(MemoryMarshal.Cast<T, byte>(span));
        }

        public void Write<T>(ReadOnlySpan<T> values)
            where T : unmanaged
        {
            _writer.Write(MemoryMarshal.Cast<T, byte>(values));
        }

        public void Write<T>(Span<T> values)
            where T : unmanaged
        {
            _writer.Write(MemoryMarshal.Cast<T, byte>(values));
        }

        public void Write<T>(IEnumerable<T> values)
            where T : unmanaged
        {
            foreach (var item in values)
                Write(item);
        }

        public int AlignPosition(int alignment)
        {
            var rem = (int)(Position % alignment);
            if (rem != 0)
            {
                var offset = alignment - rem;
                Position += offset;
                return offset;
            }

            return 0;
        }

        public void Dispose()
        {
            _writer.Dispose();
        }
    }
}
