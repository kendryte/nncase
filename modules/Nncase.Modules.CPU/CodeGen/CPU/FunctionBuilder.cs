// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
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
                var dt = tensor.ElementType;
                switch (dt)
                {
                    case var _ when dt == DataTypes.Float32:
                        WriteRdataSeg<float>(tensor, range);
                        break;
                    case var _ when dt == DataTypes.Float64:
                        WriteRdataSeg<double>(tensor, range);
                        break;
                    case var _ when dt == DataTypes.Int8:
                        WriteRdataSeg<sbyte>(tensor, range);
                        break;
                    case var _ when dt == DataTypes.Int32:
                        WriteRdataSeg<int>(tensor, range);
                        break;
                    case var _ when dt == DataTypes.Int64:
                        WriteRdataSeg<long>(tensor, range);
                        break;
                    case var _ when dt == DataTypes.UInt8:
                        WriteRdataSeg<byte>(tensor, range);
                        break;
                    case var _ when dt == DataTypes.UInt32:
                        WriteRdataSeg<uint>(tensor, range);
                        break;
                    case var _ when dt == DataTypes.UInt64:
                        WriteRdataSeg<ulong>(tensor, range);
                        break;

                    // case var _ when t == DataTypes.BFloat16:
                    //     WriteRdataSeg<BFloat16>(tensor, range);
                    //     break;
                    case var _ when dt == DataTypes.Float16:
                        WriteRdataSeg<Half>(tensor, range);
                        break;
                    case var _ when dt is VectorType vt:
                        {
                            var et = vt.ElemType;
                            switch (et)
                            {
                                case var _ when et == DataTypes.Float32:
                                    WriteVectorRdata<float>(tensor, range, vt.Lanes);
                                    break;
                                case var _ when et == DataTypes.Float64:
                                    WriteVectorRdata<double>(tensor, range, vt.Lanes);
                                    break;
                                case var _ when et == DataTypes.Int8:
                                    WriteVectorRdata<sbyte>(tensor, range, vt.Lanes);
                                    break;
                                case var _ when et == DataTypes.Int32:
                                    WriteVectorRdata<int>(tensor, range, vt.Lanes);
                                    break;
                                case var _ when et == DataTypes.Int64:
                                    WriteVectorRdata<long>(tensor, range, vt.Lanes);
                                    break;
                                case var _ when et == DataTypes.UInt8:
                                    WriteVectorRdata<byte>(tensor, range, vt.Lanes);
                                    break;
                                case var _ when et == DataTypes.UInt32:
                                    WriteVectorRdata<uint>(tensor, range, vt.Lanes);
                                    break;
                                case var _ when et == DataTypes.UInt64:
                                    WriteVectorRdata<ulong>(tensor, range, vt.Lanes);
                                    break;
                                case var _ when et == DataTypes.Float8E4M3:
                                    WriteVectorRdata<Float8E4M3>(tensor, range, vt.Lanes);
                                    break;

                                // case var _ when et == DataTypes.BFloat16:
                                //     WriteVectorRdata<BFloat16>(tensor, range, vt.Lanes);
                                //     break;
                                case var _ when et == DataTypes.Float16:
                                    WriteVectorRdata<Half>(tensor, range, vt.Lanes);
                                    break;
                                default:
                                    throw new NotSupportedException($"Not supported onnx constant data type {dt}");
                            }
                        }

                        break;
                    default:
                        throw new NotSupportedException($"Not supported onnx constant data type {dt}");
                }
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

    private void WriteRdataSeg<T>(Tensor tensor, ValueRange<ulong> range)
        where T : unmanaged, INumber<T>
    {
        var buffer = tensor.ToArray<T>().AsSpan();
        var dt = tensor.ElementType;
        var size = range.Max - range.Min;
        if ((ulong)buffer.Length * (ulong)dt.SizeInBytes != size)
        {
            throw new InvalidDataException("The Buffer Szie Not Equal!");
        }

        var chunck = 1024 * 1024 * 1024L / dt.SizeInBytes;
        int written = 0;
        long length = buffer.Length;
        _rdataWriter.Position((long)range.Min);
        while (length > 0)
        {
            var sizeToWrite = (int)System.Math.Min(length, chunck);
            _rdataWriter.Write(MemoryMarshal.Cast<T, byte>(buffer.Slice(written, sizeToWrite)));
            written += sizeToWrite;
            length -= sizeToWrite;
        }
    }

    private void WriteRdataSeg<T, TVector>(Tensor tensor, ValueRange<ulong> range)
        where T : unmanaged, INumber<T>
        where TVector : unmanaged, IEquatable<TVector>
    {
        var buffer = tensor.ToArray<TVector>().AsSpan();
        var dt = tensor.ElementType;
        var size = range.Max - range.Min;
        if ((ulong)buffer.Length * (ulong)dt.SizeInBytes != size)
        {
            throw new InvalidDataException("The Buffer Size Not Equal!");
        }

        var chunck = 1024 * 1024 * 1024L / dt.SizeInBytes;
        int written = 0;
        long length = buffer.Length;
        _rdataWriter.Position((long)range.Min);
        while (length > 0)
        {
            var sizeToWrite = (int)System.Math.Min(length, chunck);
            _rdataWriter.Write(MemoryMarshal.Cast<TVector, byte>(buffer.Slice(written, sizeToWrite)));
            written += sizeToWrite;
            length -= sizeToWrite;
        }
    }

    private void WriteVectorRdata<T>(Tensor tensor, ValueRange<ulong> range, IRArray<int> lanes)
        where T : unmanaged, INumber<T>, IEquatable<T>
    {
        switch (lanes)
        {
            case var _ when lanes.ToArray().SequenceEqual([4]):
                WriteRdataSeg<T, Vector4<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([4, 4]):
                WriteRdataSeg<T, Vector4x4<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([8]):
                WriteRdataSeg<T, Vector8<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([8, 8]):
                WriteRdataSeg<T, Vector8x8<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([16]):
                WriteRdataSeg<T, Vector16<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([16, 16]):
                WriteRdataSeg<T, Vector16x16<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([32]):
                WriteRdataSeg<T, Vector32<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([32, 16]):
                WriteRdataSeg<T, Vector32x16<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([32, 32]):
                WriteRdataSeg<T, Vector32x32<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([32, 64]):
                WriteRdataSeg<T, Vector32x64<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([64]):
                WriteRdataSeg<T, Vector64<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([128]):
                WriteRdataSeg<T, Vector128<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([32, 128]):
                WriteRdataSeg<T, Vector32x128<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([64, 32]):
                WriteRdataSeg<T, Vector64x32<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([64, 64]):
                WriteRdataSeg<T, Vector64x64<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([64, 128]):
                WriteRdataSeg<T, Vector64x128<T>>(tensor, range);
                break;
            case var _ when lanes.ToArray().SequenceEqual([128, 64]):
                WriteRdataSeg<T, Vector128x64<T>>(tensor, range);
                break;
            default:
                throw new NotSupportedException($"Not supported onnx constant vector type");
        }
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
