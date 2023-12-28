// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CodeGen.CPU;
using Nncase.IR;

namespace Nncase.CodeGen.CPU;

/// <summary>
/// StackVM function builder.
/// </summary>
internal class FunctionBuilder : IDisposable
{
    private readonly uint _id;
    private readonly SectionManager _sectionManager;
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
        _sectionManager = new();
        _textWriter = _sectionManager.GetWriter(WellknownSectionNames.Text);
        _rdataWriter = rdataWriter;
    }

    public unsafe ILinkableFunction Build(TIR.PrimFunction function)
    {
        if (function.Name.EndsWith("kernel"))
        {
            // 1. convert func to csource
            var visitor = new KernelCSourceConvertVisitor();
            visitor.Visit(function);
            var functionCSource = visitor.GetCSource();
            // 3. write the rdata
            foreach (var (@const, range) in function.SchedResult.Rdatas)
            {
                var bytes = ((TensorConst)@const).Value.BytesBuffer;
                var size = range.Max - range.Min;
                if ((uint)bytes.Length != size)
                {
                    throw new InvalidDataException("The Buffer Szie Not Equal!");
                }

                _rdataWriter.Position(range.Min);
                _rdataWriter.Write(bytes);
            }

            return new LinkableKernelFunction(_id, function, functionCSource, _sectionManager.GetContent(WellknownSectionNames.Text));
        }
        else if (function.Name.EndsWith("device"))
        {
            var visitor = new DeviceCSourceConvertVisitor();
            visitor.Visit(function);
            var header = visitor.GetHeader();

            return new LinkableDeviceFunction(_id, function, header, _sectionManager.GetContent(WellknownSectionNames.Text));
        }

        throw new NotSupportedException();
    }

    public void Dispose()
    {
    }
}
