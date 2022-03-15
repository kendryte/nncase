// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.CodeGen;

/// <summary>
/// static class for codegen collection.
/// </summary>
public static class CodeGenExtensions
{
    /// <summary>
    /// schedule and build the IRModule to RTModel.
    /// </summary>
    /// <param name="module"> input module. </param>
    /// <param name="target"> target information. </param>
    /// <returns> the runtime model instance. </returns>
    public static IRTModel ToRTModel(this IRModule module, ITarget target)
    {
        var sch = target.CreateScheduler(module);
        var schr = sch.Schedule();
        return target.CreateRTModel(schr);
    }
}

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
}
