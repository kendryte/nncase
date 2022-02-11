// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Tensorflow;
using Tensorflow.NumPy;
using Shape = Tensorflow.Shape;

namespace Nncase.Evaluator;

/// <summary>
/// TensorFlow extension.
/// </summary>
public static class TensorflowExtension
{
    public static Const ToConst(this Tensor tensor)
    {
        return new Const(
            new TensorType(ToDataType(tensor.dtype), tensor.shape.as_int_list()),

            // todo:this is copy
            tensor.BufferToArray());
    }

    public static NDArray ToTFTensor(this Const tensor)
    {
        var span = new Span<byte>(tensor.Data.Data());
        var dt = tensor.CheckedDataType.ToTFType();
        var shape = new Shape(tensor.CheckedShape.ToValueArray());
        return new NDArray(tensor.Data.Data(), shape, dt);
    }

    private static readonly Dictionary<DataType, TF_DataType> _dataTypesToTorchType = new()
    {
        { DataType.Bool, TF_DataType.TF_BOOL },
        { DataType.Int8, TF_DataType.TF_INT8 },
        { DataType.Int16, TF_DataType.TF_INT16 },
        { DataType.Int32, TF_DataType.TF_INT32 },
        { DataType.Int64, TF_DataType.TF_INT64 },
        { DataType.UInt8, TF_DataType.TF_UINT8 },
        { DataType.Float16, TF_DataType.TF_HALF },
        { DataType.Float32, TF_DataType.TF_FLOAT },
        { DataType.Float64, TF_DataType.TF_DOUBLE },
    };

    private static readonly Dictionary<TF_DataType, DataType> _TorchTypeTodataTypes = new()
    {
        { TF_DataType.TF_BOOL, DataType.Bool },
        { TF_DataType.TF_INT8, DataType.Int8 },
        { TF_DataType.TF_INT16, DataType.Int16 },
        { TF_DataType.TF_INT32, DataType.Int32 },
        { TF_DataType.TF_INT64, DataType.Int64 },
        { TF_DataType.TF_UINT8, DataType.UInt8 },
        { TF_DataType.TF_HALF, DataType.Float16 },
        { TF_DataType.TF_FLOAT, DataType.Float32 },
        { TF_DataType.TF_DOUBLE, DataType.Float64 },
    };

    public static TF_DataType ToTFType(this DataType dt) => _dataTypesToTorchType[dt];

    public static DataType ToDataType(this TF_DataType dt) => _TorchTypeTodataTypes[dt];
}
