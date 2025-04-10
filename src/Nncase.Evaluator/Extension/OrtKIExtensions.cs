﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
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
        return Tensor.From(tensor.DataType.ToDataType(), new TensorInitializerWithOrt(tensor), tensor.Shape);
    }

    public static Tensor ToTensor(this OrtKISharp.Tensor tensor, TensorType tensorType)
    {
        return Tensor.From(tensorType.DType, new TensorInitializerWithOrt(tensor), tensorType.Shape.IsFixed ? tensorType.Shape : tensor.Shape);
    }

    public static TensorValue ToValue(this OrtKISharp.Tensor tensor)
    {
        return tensor.ToTensor();
    }

    public static TensorValue ToValue(this OrtKISharp.Tensor tensor, DataType dataType) => dataType switch
    {
        VectorType vectorType => Tensor.From(
            vectorType with { ElemType = tensor.DataType.ToDataType() },
            new TensorInitializerWithOrt(tensor),
            tensor.Shape.Take(tensor.Shape.Length - vectorType.Lanes.Count).ToArray()).CastTo(vectorType),
        PrimType primType => tensor.ToTensor().CastTo(primType),
        _ => throw new NotSupportedException(),
    };

    public static OrtKISharp.Tensor ToOrtTensor(this Tensor tensor) => tensor.ElementType switch
    {
        VectorType vectorType => ToOrtTensor(tensor, vectorType.ElemType.ToOrtType(), tensor.Dimensions.ToArray().Concat(vectorType.Lanes.Select(x => (long)x)).ToArray()),
        PrimType primType => ToOrtTensor(tensor, primType.ToOrtType(), tensor.Dimensions.ToArray()),
        _ => throw new NotSupportedException(),
    };

    public static OrtKISharp.Tensor ScalarToOrtTensor(this Tensor tensor)
    {
        if (!tensor.Shape.IsScalar)
        {
            throw new InvalidOperationException("Tensor is not a scala in ScalarToOrtTensor");
        }

        return ToOrtTensor(tensor, tensor.ElementType.ToOrtType(), [1]);
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

    private static OrtKISharp.Tensor ToOrtTensor(Tensor tensor, OrtDataType ortDataType, long[] shape)
    {
        return OrtKISharp.Tensor.MakeTensor(tensor.PinBuffer(), ortDataType, shape);
    }

    private sealed class TensorInitializerWithOrt : ITensorInitializer
    {
        private static readonly MethodInfo _initializeUnmanagedFunc = typeof(TensorInitializerWithOrt)
            .GetMethod(nameof(InitializeUnmanaged), BindingFlags.NonPublic | BindingFlags.Instance)!;

        private readonly OrtKISharp.Tensor _tensor;

        public TensorInitializerWithOrt(OrtKISharp.Tensor tensor)
        {
            _tensor = tensor;
        }

        public void Initialize<T>(Tensor<T> tensor)
            where T : struct, IEquatable<T>
        {
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
            {
                throw new NotSupportedException("Tensor<T> with reference type is not supported.");
            }

            _initializeUnmanagedFunc.MakeGenericMethod(typeof(T)).Invoke(this, [tensor]);
        }

        private void InitializeUnmanaged<T>(Tensor<T> tensor)
            where T : unmanaged, IEquatable<T>
        {
            var span = MemoryMarshal.Cast<T, byte>(tensor.Buffer.Span);
            _tensor.GetBuffer<byte>().CopyTo(span);
        }
    }
}
