// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.Targets;
using Nncase.Utilities;

namespace Nncase.CodeGen.NTT;

/// <summary>
/// StackVM function builder.
/// </summary>
internal class FunctionBuilder
{
    private readonly uint _id;
    private readonly SectionManager _sectionManager;
    private readonly BinaryWriter _textWriter;
    private readonly BinaryWriter _rdataWriter;
    private readonly IReadOnlyList<BinaryWriter> _threadLocalRdataWriters;
    private readonly IReadOnlyList<BinaryWriter> _blockLocalRdataWriters;

    public FunctionBuilder(uint id, BinaryWriter rdataWriter, IReadOnlyList<BinaryWriter> threadLocalRdataWriters, IReadOnlyList<BinaryWriter> blockLocalRdataWriters, Targets.NTTTargetOptions targetOptions)
    {
        _id = id;
        _sectionManager = new();
        _textWriter = _sectionManager.GetWriter(WellknownSectionNames.Text);
        _rdataWriter = rdataWriter;
        _threadLocalRdataWriters = threadLocalRdataWriters;
        _blockLocalRdataWriters = blockLocalRdataWriters;
        TargetOptions = targetOptions;
    }

    public NTTTargetOptions TargetOptions { get; }

    public unsafe ILinkableFunction Build(TIR.PrimFunction function)
    {
        if (!function.Name.Contains("device_func", StringComparison.Ordinal))
        {
            // 1. write the rdata
            ulong rdataPoolSize = ulong.MinValue;
            foreach (var (@const, range) in function.SchedResult.Rdatas)
            {
                var tensor = ((TensorConst)@const).Value;
                var size = range.Max - range.Min;
                rdataPoolSize = System.Math.Max(range.Max, rdataPoolSize);
                if ((ulong)tensor.Length * (ulong)tensor.ElementType.SizeInBytes != size)
                {
                    throw new InvalidDataException("The Buffer Size Not Equal!");
                }

                _rdataWriter.Position(checked((long)range.Min));
                tensor.Serialize(_rdataWriter.BaseStream);
            }

            // 2. write the thread local rdata
            ulong threadLocalRdataPoolSize = ulong.MinValue;
            foreach (var (@const, range) in function.SchedResult.ThreadLocalRdatas)
            {
                var tensor = ((TensorConst)@const).Value;
                var distributedType = (DistributedType)@const.CheckedType;
                var size = range.Max - range.Min;
                threadLocalRdataPoolSize = System.Math.Max(range.Max, threadLocalRdataPoolSize);
                var dividedDims = DistributedUtility.GetDividedTensorType(distributedType).Shape.ToValueArray();
                var localStrides = TensorUtilities.GetDefaultStrides(dividedDims);
                for (int i = 0; i < _threadLocalRdataWriters.Count; i++)
                {
                    var threadLocalRdataWriter = _threadLocalRdataWriters[i];
                    var shardIndex = DistributedUtility.GetUnraveledIndex(i, TargetOptions.Hierarchies[0]);
                    (var localOffset, var localShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                    var linearOffset = TensorUtilities.GetLinearOffset(tensor.Strides, localOffset);

                    if ((ulong)TensorUtilities.GetProduct(localShape) * (ulong)tensor.ElementType.SizeInBytes > size)
                    {
                        throw new InvalidDataException("The Buffer Size Not Equal!");
                    }

                    threadLocalRdataWriter.Position(checked((long)range.Min));
                    tensor.Serialize(threadLocalRdataWriter.BaseStream, linearOffset, localShape, localStrides);
                }
            }

            // 2. write the block local rdata
            ulong blockLocalRdataPoolSize = ulong.MinValue;
            foreach (var (@const, range) in function.SchedResult.BlockLocalRdatas)
            {
                var tensor = ((TensorConst)@const).Value;
                var distributedType = (DistributedType)@const.CheckedType;
                var size = range.Max - range.Min;
                blockLocalRdataPoolSize = System.Math.Max(range.Max, blockLocalRdataPoolSize);
                var dividedDims = DistributedUtility.GetDividedTensorType(distributedType).Shape.ToValueArray();
                var localStrides = TensorUtilities.GetDefaultStrides(dividedDims);
                for (int i = 0; i < _blockLocalRdataWriters.Count; i++)
                {
                    var blockLocalRdataWriter = _blockLocalRdataWriters[i];
                    var shardIndex = DistributedUtility.GetUnraveledIndex(i, TargetOptions.Hierarchies[0]);
                    (var localOffset, var localShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                    var linearOffset = TensorUtilities.GetLinearOffset(tensor.Strides, localOffset);

                    if ((ulong)TensorUtilities.GetProduct(localShape) * (ulong)tensor.ElementType.SizeInBytes > size)
                    {
                        throw new InvalidDataException("The Buffer Size Not Equal!");
                    }

                    blockLocalRdataWriter.Position(checked((long)range.Min));
                    tensor.Serialize(blockLocalRdataWriter.BaseStream, linearOffset, localShape, localStrides);
                }
            }

            // 4. build function.
            var visitor = new KernelCSourceConvertVisitor(function.SchedResult.DataAlign, function.SchedResult.DataUsage, rdataPoolSize, threadLocalRdataPoolSize, blockLocalRdataPoolSize, TargetOptions);
            visitor.Visit(function);
            var functionCSource = visitor.GetCSource();

            // 5. write the kernel desc
            using (var writer = _sectionManager.GetWriter(LinkableKernelFunction.KernelHeaderSectionName))
            {
                var header = default(KernelDescHeader);
                header.OutputAlign = (uint)function.SchedResult.OutputAlign;
                header.LocalDataAlign = (uint)function.SchedResult.DataAlign;
                header.OutputPoolSize = function.SchedResult.OutputUsage;
                header.LocalDataPoolSize = function.SchedResult.DataUsage;
                writer.Write(ref header);
            }

            var kernelDescSection = new LinkedSection(_sectionManager.GetContent(LinkableKernelFunction.KernelHeaderSectionName)!, ".desc", 0, 8, (uint)sizeof(KernelDescHeader));
            return new LinkableKernelFunction(_id, function, functionCSource, _sectionManager.GetContent(WellknownSectionNames.Text)!, kernelDescSection);
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
}
