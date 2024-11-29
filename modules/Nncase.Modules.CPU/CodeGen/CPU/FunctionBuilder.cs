// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Drawing;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CodeGen.CPU;
using Nncase.IR;
using Nncase.Targets;

namespace Nncase.CodeGen.CPU;

/// <summary>
/// StackVM function builder.
/// </summary>
internal class FunctionBuilder
{
    public const string KernelHeaderSectionName = ".desc";
    private readonly uint _id;
    private readonly SectionManager _sectionManager;
    private readonly BinaryWriter _textWriter;
    private readonly BinaryWriter _rdataWriter;

    public FunctionBuilder(uint id, BinaryWriter rdataWriter, Targets.CpuTargetOptions targetOptions)
    {
        _id = id;
        _sectionManager = new();
        _textWriter = _sectionManager.GetWriter(WellknownSectionNames.Text);
        _rdataWriter = rdataWriter;
        TargetOptions = targetOptions;
    }

    public CpuTargetOptions TargetOptions { get; }

    public unsafe ILinkableFunction Build(TIR.PrimFunction function)
    {
        if (function.Name.EndsWith("kernel"))
        {
            // 1. write the kernel header
            using (var writer = _sectionManager.GetWriter(KernelHeaderSectionName))
            {
                var header = default(DescHeader);
                header.ThreadDim = (uint)TargetOptions.Hierarchies[0][^1];
                header.BlockDim = TargetOptions.Hierarchies[0].Length < 2 ? 1 : (uint)TargetOptions.Hierarchies[0][^2];
                header.ChipDim = TargetOptions.Hierarchies[0].Length < 3 ? 1 : (uint)TargetOptions.Hierarchies[0][^3];
                writer.Write(ref header);
            }

            // 2. write the rdata
            ulong rdataPoolSize = ulong.MinValue;
            foreach (var (@const, range) in function.SchedResult.Rdatas)
            {
                var tensor = ((TensorConst)@const).Value;
                _rdataWriter.Position(checked((long)range.Min));
                var size = range.Max - range.Min;
                if ((ulong)tensor.Length * (ulong)tensor.ElementType.SizeInBytes != size)
                {
                    throw new InvalidDataException("The Buffer Size Not Equal!");
                }

                tensor.Serialize(_rdataWriter.BaseStream);
            }

            // 3. build function.
            var visitor = new KernelCSourceConvertVisitor(function.SchedResult.DataAlign, function.SchedResult.DataUsage, rdataPoolSize, TargetOptions);
            visitor.Visit(function);
            var functionCSource = visitor.GetCSource();

            return new LinkableKernelFunction(_id, function, functionCSource, _sectionManager.GetContent(WellknownSectionNames.Text)!, new LinkedSection(_sectionManager.GetContent(KernelHeaderSectionName), KernelHeaderSectionName, 0, 8, (uint)sizeof(DescHeader)));
        }
        else
        {
            var visitor = new DeviceCSourceConvertVisitor();
            visitor.Visit(function);
            var header = visitor.GetHeader();
            return new LinkableDeviceFunction(_id, function, header, _sectionManager.GetContent(WellknownSectionNames.Text)!);
        }

        throw new NotSupportedException("the function name is invalid");
    }

    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct DescHeader
    {
        [MarshalAs(UnmanagedType.U4)]
        public uint ThreadDim;

        [MarshalAs(UnmanagedType.U4)]
        public uint BlockDim;

        [MarshalAs(UnmanagedType.U4)]
        public uint ChipDim;

        [MarshalAs(UnmanagedType.U4)]
        public uint Reserved0;
    }
}
