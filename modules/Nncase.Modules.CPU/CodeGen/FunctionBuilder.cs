// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.CodeGen.CPU;

/// <summary>
/// StackVM function builder.
/// </summary>
internal class FunctionBuilder : IDisposable
{
    private readonly uint _id;
    private readonly MemoryStream _textContent = new MemoryStream();
    private readonly BinaryWriter _textWriter;
    private readonly BinaryWriter _rdataWriter;

    /// <summary>
    /// NOTE sync with the cpu runtime function.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryRange
    {
        public uint Start;
        public uint Size;
    }

    /// <summary>
    /// NOTE sync with the cpu runtime function.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct DescHeader
    {
        /// <summary>
        /// input pool size.
        /// </summary>
        public uint InputPoolSize;

        /// <summary>
        /// output pool size.
        /// </summary>
        public uint OutputPoolSize;

        /// <summary>
        /// input numbers.
        /// </summary>
        public uint Inputs;

        /// <summary>
        /// output numbers.
        /// </summary>
        public uint Outputs;
    }

    public FunctionBuilder(uint id, BinaryWriter rdataWriter)
    {
        _id = id;
        _textWriter = new BinaryWriter(_textContent, Encoding.UTF8, leaveOpen: true);
        _rdataWriter = rdataWriter;
    }

    public unsafe LinkableFunction Build(TIR.PrimFunction function)
    {
        // 1. write the inst
        // new InstSerializeVisitor(_textWriter).Visit(function.Body);

        // 2. write the desc
        var descContent = new MemoryStream();
        using (var descWriter = new BinaryWriter(descContent, Encoding.UTF8))
        {
            DescHeader header = new() { InputPoolSize = 0, OutputPoolSize = 0, Inputs = 0, Outputs = 0 };
            long headerStart = descWriter.Position();
            descWriter.Skip((ulong)sizeof(DescHeader));

            foreach (var input in function.Parameters.AsValueEnumerable()
                                .Where(buf => buf.MemLocation == Schedule.MemoryLocation.Input))
            {
                header.Inputs++;
                var rg = new MemoryRange { Start = checked((uint)input.Start), Size = checked((uint)input.Size) };
                descWriter.Write(ref rg);
                header.InputPoolSize = Math.Max(header.InputPoolSize, rg.Start + rg.Size);
            }

            foreach (var output in function.Parameters.AsValueEnumerable().Where(buf => buf.MemLocation == Schedule.MemoryLocation.Output))
            {
                header.Outputs++;
                var rg = new MemoryRange { Start = checked((uint)output.Start), Size = checked((uint)output.Size) };
                descWriter.Write(ref rg);
                header.OutputPoolSize = Math.Max(header.OutputPoolSize, rg.Start + rg.Size);
            }

            descWriter.Position(headerStart);
            descWriter.Write(ref header);
        }

        // 3. write the rdata
        foreach (var buffer in function.SchedResult.Rdatas)
        {
            var bytes = buffer.Const!.Value.BytesBuffer;
            if ((uint)bytes.Length != buffer.Size)
            {
                throw new InvalidDataException("The Buffer Szie Not Equal!");
            }

            _rdataWriter.Position((uint)buffer.Start);
            _rdataWriter.Write(bytes);
        }

        return new LinkableFunction(_id, function, _textContent.ToArray(), descContent.ToArray());
    }

    public void Dispose()
    {
    }
}
