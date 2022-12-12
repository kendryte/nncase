using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Nncase.IR;
using OrtKISharp;

namespace Nncase.Evaluator;

/// <summary>
/// OrtKI extensions.
/// </summary>
public static class OrtKIExtensions
{
    private static readonly Dictionary<DataType, OrtDataType> _dataTypesToOrtType = new()
    {
        { DataTypes.Boolean, OrtDataType.Bool },
        { DataTypes.Int8, OrtDataType.Int8 },
        { DataTypes.Int16, OrtDataType.Int16 },
        { DataTypes.Int32, OrtDataType.Int32 },
        { DataTypes.Int64, OrtDataType.Int64 },
        { DataTypes.UInt8, OrtDataType.UInt8 },
        { DataTypes.UInt16, OrtDataType.UInt16 },
        { DataTypes.UInt32, OrtDataType.UInt32 },
        { DataTypes.UInt64, OrtDataType.UInt64 },
        { DataTypes.BFloat16, OrtDataType.BFloat16 },
        { DataTypes.Float16, OrtDataType.Float16 },
        { DataTypes.Float32, OrtDataType.Float },
        { DataTypes.Float64, OrtDataType.Double },
    };

    private static readonly Dictionary<OrtDataType, DataType> _ortTypeTodataTypes = new()
    {
        { OrtDataType.Bool, DataTypes.Boolean },
        { OrtDataType.Int8, DataTypes.Int8 },
        { OrtDataType.Int16, DataTypes.Int16 },
        { OrtDataType.Int32, DataTypes.Int32 },
        { OrtDataType.Int64, DataTypes.Int64 },
        { OrtDataType.UInt8, DataTypes.UInt8 },
        { OrtDataType.UInt16, DataTypes.UInt16 },
        { OrtDataType.UInt32, DataTypes.UInt32 },
        { OrtDataType.UInt64, DataTypes.UInt64 },
        { OrtDataType.BFloat16, DataTypes.BFloat16 },
        { OrtDataType.Float16, DataTypes.Float16 },
        { OrtDataType.Float, DataTypes.Float32 },
        { OrtDataType.Double, DataTypes.Float64 },
    };

    public static Tensor ToTensor(this OrtKISharp.Tensor tensor)
    {
        var mmgr = new OrtTensorMemoryManager(tensor);
        return Tensor.FromBytes(tensor.DataType.ToDataType(), mmgr.Memory, tensor.Shape.ToInts());
    }

    public static TensorValue ToValue(this OrtKISharp.Tensor tensor)
    {
        return tensor.ToTensor();
    }

    public static OrtKISharp.Tensor ToOrtTensor(this Tensor tensor)
    {
        var shape = tensor.Dimensions.ToArray();
        return tensor.ToOrtTensor(shape);
    }

    public static OrtKISharp.Tensor ScalarToOrtTensor(this Tensor tensor)
    {
        if (!tensor.Shape.IsScalar)
        {
            throw new InvalidOperationException("Tensor is not a scala in ScalarToOrtTensor");
        }

        return tensor.ToOrtTensor(new[] { 1 });
    }

    public static OrtDataType ToOrtType(this DataType dt)
    {
        if (_dataTypesToOrtType.TryGetValue(dt, out var type))
        {
            return type;
        }

        throw new ArgumentOutOfRangeException("Unsupported DataType: " + dt);
    }

    public static DataType ToDataType(this OrtDataType dt)
    {
        if (_ortTypeTodataTypes.TryGetValue(dt, out var type))
        {
            return type;
        }

        throw new ArgumentOutOfRangeException("Unsupported OrtDataType: " + dt);
    }

    private static OrtKISharp.Tensor ToOrtTensor(this Tensor tensor, int[] shape)
    {
        return OrtKISharp.Tensor.MakeTensor(tensor.PinBuffer(), tensor.ElementType.ToOrtType(), shape.ToLongs());
    }

    private sealed class OrtTensorMemoryManager : MemoryManager<byte>
    {
        private readonly OrtKISharp.Tensor _tensor;

        public OrtTensorMemoryManager(OrtKISharp.Tensor tensor)
        {
            _tensor = tensor;
        }

        public override Span<byte> GetSpan()
        {
            return _tensor.BytesBuffer;
        }

        public unsafe override MemoryHandle Pin(int elementIndex = 0)
        {
            var basePtr = Unsafe.AsPointer(ref _tensor.BytesBuffer.GetPinnableReference());
            var pointer = Unsafe.Add<byte>(basePtr, elementIndex);
            return new MemoryHandle(pointer, pinnable: this);
        }

        public override void Unpin()
        {
        }

        protected override void Dispose(bool disposing)
        {
        }
    }
}
