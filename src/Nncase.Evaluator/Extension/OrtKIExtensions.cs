// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
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
        return Tensor.FromBytes(tensor.DataType.ToDataType(), tensor.BytesBuffer.ToArray(), tensor.Shape.ToInts());
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

    public static OrtKISharp.Tensor BroadcastTo(this OrtKISharp.Tensor tensor, long[] shape, OrtDataType dtype = OrtDataType.Float) => tensor + OrtKISharp.Tensor.Empty(shape, dtype);

    public static OrtKISharp.Tensor Pack(this OrtKISharp.Tensor tensor, int lanes, int axis)
    {
        if (axis < 0)
        {
            return tensor;
        }

        var shape = tensor.Shape;
        var dividedShape = shape.Take(axis).Concat(new[] { shape[axis] / lanes, lanes }).Concat(shape.Skip(axis + 1)).ToArray();
        var perm = Enumerable.Range(0, axis + 1).Concat(Enumerable.Range(axis + 2, dividedShape.Length - (axis + 2))).Concat(new[] { axis + 1 }).Select(i => (long)i).ToArray();
        return OrtKI.Transpose(OrtKI.Reshape(tensor, dividedShape, 0), perm);
    }

    public static OrtKISharp.Tensor Unpack(this OrtKISharp.Tensor tensor, int axis)
    {
        var perm = Enumerable.Range(0, tensor.Shape.Length);
        perm = perm.Take(axis + 1).Concat(new[] { perm.Last() }).Concat(perm.Skip(axis + 1).SkipLast(1));
        var unpacked = OrtKI.Transpose(tensor, perm.Select(i => (long)i).ToArray());
        var shape = unpacked.Shape.ToList();
        shape[axis] = shape[axis] * shape[axis + 1];
        shape.RemoveAt(axis + 1);
        return OrtKI.Reshape(unpacked, shape.ToArray(), 0);
    }

    private static OrtKISharp.Tensor ToOrtTensor(this Tensor tensor, int[] shape)
    {
        return OrtKISharp.Tensor.MakeTensor(tensor.PinBuffer(), tensor.ElementType.ToOrtType(), shape.ToLongs());
    }
}
