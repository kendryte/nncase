using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Runtime
{
    public ref struct MemoryReader
    {
        private ReadOnlyMemory<byte> _memory;

        public bool IsEmpty => _memory.IsEmpty;

        public MemoryReader(ReadOnlyMemory<byte> span)
        {
            _memory = span;
        }

        public ref readonly T As<T>()
            where T : unmanaged
        {
            return ref MemoryMarshal.AsRef<T>(_memory.Span);
        }

        public T Read<T>()
            where T : unmanaged
        {
            var value = As<T>();
            Advance(Unsafe.SizeOf<T>());
            return value;
        }

        public void Read<T>(out T value)
            where T : unmanaged
        {
            value = As<T>();
            Advance(Unsafe.SizeOf<T>());
        }

        public ReadOnlySpan<T> ReadSpan<T>(int count)
            where T : unmanaged
        {
            var size = Unsafe.SizeOf<T>() * count;
            var subspan = MemoryMarshal.Cast<byte, T>(_memory.Slice(0, size).Span);
            Advance(size);
            return subspan;
        }

        public T[] ReadArray<T>(int count)
            where T : unmanaged
        {
            var size = Unsafe.SizeOf<T>() * count;
            var subspan = MemoryMarshal.Cast<byte, T>(_memory.Slice(0, size).Span);
            Advance(size);
            return subspan.ToArray();
        }

        private void Advance(int size)
        {
            _memory = _memory.Slice(size);
        }
    }
}
