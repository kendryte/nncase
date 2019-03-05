using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Text;

namespace NnCase.Converter
{
    public static class BinaryWriterExtensions
    {
        public static void WriteAsBytes(this BinaryWriter bw, byte[] value)
        {
            bw.Write(value);
        }

        public static void WriteAsUInt8(this BinaryWriter bw, byte value)
        {
            bw.Write(value);
        }

        public static void WriteAsUInt16(this BinaryWriter bw, ushort value)
        {
            bw.Write(value.ToBigEndian());
        }

        public static void WriteAsUInt32(this BinaryWriter bw, uint value)
        {
            bw.Write(value.ToBigEndian());
        }

        public static void WriteAsIPAddress(this BinaryWriter bw, IPAddress value)
        {
            bw.Write(value.GetAddressBytes());
        }

        public static void WriteAsString(this BinaryWriter bw, string value)
        {
            var bytes = Encoding.UTF8.GetBytes(value);
            Debug.Assert(bytes.Length <= ushort.MaxValue);
            bw.WriteAsUInt16((ushort)bytes.Length);
            bw.WriteAsBytes(bytes);
        }

        public static void WriteAsKey(this BinaryWriter bw, byte[] value)
        {
            Debug.Assert(value.Length <= ushort.MaxValue);
            bw.WriteAsUInt16((ushort)value.Length);
            bw.WriteAsBytes(value);
        }
    }

    public static class DataTypeSizeExtensions
    {
        public static uint SizeOfVarInt(this uint value)
        {
            uint numWrite = 0;
            do
            {
                value >>= 7;
                numWrite++;
            }
            while (value != 0);
            return numWrite;
        }

        public static ushort ToBigEndian(this ushort value)
        {
            return (ushort)((value >> 8) | (((byte)value) << 8));
        }

        public static uint ToBigEndian(this uint value)
        {
            return (value >> 24) | ((value & 0x00FF_0000) >> 8) |
                ((value & 0x0000_FF00) << 8) | ((value & 0x0000_00FF) << 24);
        }

        public static ulong ToBigEndian(this ulong value)
        {
            return (value >> 56) | ((value & 0x00FF_0000_0000_0000) >> 40) | ((value & 0x0000_FF00_0000_0000) >> 24) |
                ((value & 0x0000_00FF_0000_0000) >> 8) | ((value & 0x0000_0000_FF00_0000) << 8) | ((value & 0x0000_0000_00FF_0000) << 24) |
                ((value & 0x0000_0000_0000_FF00) << 40) | ((value & 0x0000_0000_0000_00FF) << 56);
        }

        public static int ToVarInt(this ReadOnlySpan<byte> data)
        {
            int result = 0;
            for (int i = 0; i < data.Length; i++)
                result = (result << 8) | data[i];
            return result;
        }
    }
}
