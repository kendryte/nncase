using Nncase.IR;
namespace Nncase.CodeGen;

/// <summary>
/// static class for codegen collection
/// </summary>
public static class CodeGenExtension
{
    /// <summary>
    /// schedule and build the IRModule to RTModel
    /// </summary>
    /// <param name="mod"> input module </param>
    /// <param name="target"> target information </param>
    /// <returns> the runtime model instance </returns>
    public static IRTModel ToRTModel(this IRModule mod, ITarget target)
    {
        var sch = target.CreateScheduler(mod);
        var schr = sch.Schedule();
        return target.CreateRTModel(schr);
    }
}

/// <summary>
/// BinaryWriterExtension
/// </summary>
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

    /// <summary>
    /// get current position
    /// </summary>
    /// <param name="writer"></param>
    /// <returns></returns>
    public static long Position(this BinaryWriter writer)
    {
        return writer.BaseStream.Position;
    }

    /// <summary>
    /// move the wirter to target position
    /// </summary>
    /// <param name="writer"></param>
    /// <param name="pos"></param>
    /// <returns></returns>
    public static long Position(this BinaryWriter writer, long pos)
    {
        return writer.Seek((int)pos, SeekOrigin.Begin);
    }

    /// <summary>
    /// skip the length
    /// </summary>
    /// <param name="writer"></param>
    /// <param name="len"></param>
    public static void Skip(this BinaryWriter writer, ulong len)
    {
        writer.Seek((int)len, SeekOrigin.Current);
    }

}
