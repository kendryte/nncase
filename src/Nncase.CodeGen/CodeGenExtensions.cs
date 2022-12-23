// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using Nncase.IR;

namespace Nncase.CodeGen;

/// <summary>
/// BinaryWriterExtension.
/// </summary>
public static class BinaryWriterExtensions
{
    /// <summary>
    /// write the byte 0 into the stream.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <param name="alignment">The desired alignment.</param>
    /// <returns>Padded bytes.</returns>
    public static long AlignPosition(this BinaryWriter writer, long alignment)
    {
        var pos = writer.Position();
        var rem = pos % alignment;
        if (rem != 0)
        {
            var off = alignment - rem;
            for (int i = 0; i < off; i++)
            {
                writer.Write((byte)0);
            }

            return off;
        }

        return 0;
    }

    /// <summary>
    /// Get current position.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <returns>The current position.</returns>
    public static long Position(this BinaryWriter writer)
    {
        writer.Flush();
        return writer.BaseStream.Position;
    }

    /// <summary>
    /// Set current position.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <param name="pos">The desired position.</param>
    /// <returns>The current position.</returns>
    public static long Position(this BinaryWriter writer, long pos)
    {
        return writer.Seek((int)pos, SeekOrigin.Begin);
    }

    /// <summary>
    /// Skip bytes of length.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <param name="len">Bytes to skip.</param>
    public static void Skip(this BinaryWriter writer, ulong len)
    {
        writer.Seek((int)len, SeekOrigin.Current);
    }

    public static unsafe void Write<T>(this BinaryWriter writer, ref T value)
        where T : unmanaged
    {
        var span = MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref value, 1));
        writer.Write(span);
    }

    public static void WriteByLength(this BinaryWriter writer, long value, int length)
    {
        switch (length)
        {
            case 1:
                writer.Write(checked((byte)value));
                break;
            case 2:
                writer.Write(checked((ushort)value));
                break;
            case 4:
                writer.Write(checked((uint)value));
                break;
            case 8:
                writer.Write(checked((ulong)value));
                break;
            default:
                throw new ArgumentException("Unsupported value length.");
        }
    }
}
