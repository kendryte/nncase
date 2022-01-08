using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.CodeGen.KModel;


public static class BinaryWriterExtension
{
    /// <summary>
    /// write the byte 0 into the stream.
    /// </summary>
    /// <param name="writer"></param>
    /// <param name="alignment"></param>
    /// <returns></returns>
    public static long AlignPosition(this BinaryWriter writer, long alignment)
    {
        var pos = writer.BaseStream.Position;
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


    public static long Position(this BinaryWriter writer)
    {
        return writer.BaseStream.Position;
    }

    public static long Position(this BinaryWriter writer, long pos)
    {
        return writer.Seek((int)pos, SeekOrigin.Begin);
    }

    public static void Skip(this BinaryWriter writer, ulong len)
    {
        writer.Seek((int)len, SeekOrigin.Current);
    }

}
