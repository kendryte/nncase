using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase
{
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    public unsafe struct Scalar
    {
        public const int MaxValueSize = 8;

        public DataType Type { get; }

        private fixed byte _value[MaxValueSize];

        public ref byte this[int index]
        {
            get { return ref _value[index]; }
        }

        public Scalar(float value)
        {
            Type = DataType.Float32;
            As<float>() = value;
        }

        public Scalar(byte value)
        {
            Type = DataType.UInt8;
            As<byte>() = value;
        }

        public static implicit operator Scalar(float value)
        {
            return new Scalar(value);
        }

        public static implicit operator Scalar(byte value)
        {
            return new Scalar(value);
        }

        public ref T As<T>()
            where T : unmanaged
        {
            ref var r = ref _value[0];
            return ref Unsafe.As<byte, T>(ref r);
        }
    }
}
