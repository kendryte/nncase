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
    private readonly IReadOnlyList<BinaryWriter> _localRdataWriters;

    public FunctionBuilder(uint id, BinaryWriter rdataWriter, IReadOnlyList<BinaryWriter> localRdataWriters, Targets.NTTTargetOptions targetOptions)
    {
        _id = id;
        _sectionManager = new();
        _textWriter = _sectionManager.GetWriter(WellknownSectionNames.Text);
        _rdataWriter = rdataWriter;
        _localRdataWriters = localRdataWriters;
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

            // 2. write the local rdata
            ulong localRdataPoolSize = ulong.MinValue;
            foreach (var (@const, range) in function.SchedResult.LocalRdatas)
            {
                var tensor = ((TensorConst)@const).Value;
                var distributedType = (DistributedType)@const.CheckedType;
                var size = range.Max - range.Min;
                localRdataPoolSize = System.Math.Max(range.Max, localRdataPoolSize);
                var dividedDims = DistributedUtility.GetDividedTensorType(distributedType).Shape.ToValueArray();
                var localStrides = TensorUtilities.GetStrides(dividedDims);
                for (int i = 0; i < _localRdataWriters.Count; i++)
                {
                    var localRdataWriter = _localRdataWriters[i];
                    var shardIndex = DistributedUtility.GetUnraveledIndex(i, TargetOptions.Hierarchies[0]);
                    (var localOffset, var localShape) = DistributedUtility.GetLocalOffsetAndShape(distributedType, shardIndex);
                    var linearOffset = TensorUtilities.GetIndex(tensor.Strides, localOffset);

                    if ((ulong)TensorUtilities.GetProduct(localShape) * (ulong)tensor.ElementType.SizeInBytes > size)
                    {
                        throw new InvalidDataException("The Buffer Size Not Equal!");
                    }

                    localRdataWriter.Position(checked((long)range.Min));
                    tensor.Serialize(localRdataWriter.BaseStream, linearOffset, localShape, localStrides);
                }
            }

            // 3. build function.
            var visitor = new KernelCSourceConvertVisitor(function.SchedResult.DataAlign, function.SchedResult.DataUsage, rdataPoolSize, localRdataPoolSize, TargetOptions);
            visitor.Visit(function);
            var functionCSource = visitor.GetCSource();

            return new LinkableKernelFunction(_id, function, functionCSource, _sectionManager.GetContent(WellknownSectionNames.Text)!);
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
