using System;
using System.Collections.Generic;
using Nncase.IR;
using OrtKISharp;

namespace Nncase.Evaluator;

public static class OrtKIExtension
{
    public static Tensor ToTensor(this OrtKISharp.Tensor tensor)
    {
        return Tensor.FromBytes(
            tensor.DataType.ToDataType(),
            tensor.BufferToArray(),
            tensor.Shape);
    }
    
    public static TensorValue ToValue(this OrtKISharp.Tensor tensor)
    {
        return tensor.ToTensor();
    }

    public static OrtKISharp.Tensor ToOrtTensor(this Tensor tensor)
    {
        return new OrtKISharp.Tensor(
            tensor.BytesBuffer, 
            tensor.ElementType.ToOrtType(), 
            tensor.Dimensions.ToArray());
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
        if (_OrtTypeTodataTypes.TryGetValue(dt, out var type))
        {
            return type;
        }
        throw new ArgumentOutOfRangeException("Unsupported OrtDataType: " + dt);
    }
    
    private static readonly Dictionary<DataType, OrtDataType> _dataTypesToOrtType = new()
    {
        { DataType.Bool, OrtDataType.Bool },
        { DataType.Int8, OrtDataType.Int8 },
        { DataType.Int16, OrtDataType.Int16 },
        { DataType.Int32, OrtDataType.Int32 },
        { DataType.Int64, OrtDataType.Int64 },
        { DataType.UInt8, OrtDataType.UInt8 },
        { DataType.Float16, OrtDataType.Float16 },
        { DataType.Float32, OrtDataType.Float },
        { DataType.Float64, OrtDataType.Double },
    };
    
    private static readonly Dictionary<OrtDataType, DataType> _OrtTypeTodataTypes = new()
    {
        { OrtDataType.Bool, DataType.Bool },
        { OrtDataType.Int8, DataType.Int8 },
        { OrtDataType.Int16, DataType.Int16 },
        { OrtDataType.Int32, DataType.Int32 },
        { OrtDataType.Int64, DataType.Int64 },
        { OrtDataType.UInt8, DataType.UInt8 },
        { OrtDataType.Float16, DataType.Float16 },
        { OrtDataType.Float, DataType.Float32 },
        { OrtDataType.Double, DataType.Float64 },
    };
}