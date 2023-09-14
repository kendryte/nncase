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
    /// NOTE sync with the k230 runtime function.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryRange
    {
        public uint Start;
        public uint Size;
    }

    /// <summary>
    /// NOTE sync with the k230 runtime function.
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
        // 1. convert func to csource
        var visitor = new CSourceConvertVisitor();
        visitor.Visit(function);
        var functionCSource = visitor.GetFunctionCSource();

        // 3. write the rdata
        foreach (var (@const, range) in function.SchedResult.Rdatas)
        {
            var bytes = ((TensorConst)@const).Value.BytesBuffer;
            var size = range.End.Value - range.Start.Value;
            if ((uint)bytes.Length != size)
            {
                throw new InvalidDataException("The Buffer Szie Not Equal!");
            }

            _rdataWriter.Position(range.Start.Value);
            _rdataWriter.Write(bytes);
        }

        return new LinkableFunction(_id, function, functionCSource);
    }

    public void Dispose()
    {
    }
}
