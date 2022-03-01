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
        var shape = tensor.Dimensions.IsEmpty ? new[] {1} : tensor.Dimensions.ToArray();
            return new OrtKISharp.Tensor(
            tensor.BytesBuffer, 
            tensor.ElementType.ToOrtType(), 
            shape);
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
        { DataTypes.Boolean, OrtDataType.Bool },
        { DataTypes.Int8, OrtDataType.Int8 },
        { DataTypes.Int16, OrtDataType.Int16 },
        { DataTypes.Int32, OrtDataType.Int32 },
        { DataTypes.Int64, OrtDataType.Int64 },
        { DataTypes.UInt8, OrtDataType.UInt8 },
        { DataTypes.Float16, OrtDataType.Float16 },
        { DataTypes.Float32, OrtDataType.Float },
        { DataTypes.Float64, OrtDataType.Double },
    };
    
    private static readonly Dictionary<OrtDataType, DataType> _OrtTypeTodataTypes = new()
    {
        { OrtDataType.Bool, DataTypes.Boolean },
        { OrtDataType.Int8, DataTypes.Int8 },
        { OrtDataType.Int16, DataTypes.Int16 },
        { OrtDataType.Int32, DataTypes.Int32 },
        { OrtDataType.Int64, DataTypes.Int64 },
        { OrtDataType.UInt8, DataTypes.UInt8 },
        { OrtDataType.Float16, DataTypes.Float16 },
        { OrtDataType.Float, DataTypes.Float32 },
        { OrtDataType.Double, DataTypes.Float64 },
    };
}