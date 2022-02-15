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
    /// <summary>
    /// Convert <see cref="Tensorflow.Tensor"/> to <see cref="Tensor"/>.
    /// </summary>
    /// <param name="tensor">Tensorflow tensor.</param>
    /// <returns>Converted tensor.</returns>
    public static Tensor ToTensor(this Tensorflow.Tensor tensor)
    {
        // TODO: Copy-free
        return Tensor.FromBytes(ToDataType(tensor.dtype), tensor.BufferToArray(), tensor.shape.as_int_list());
    }

    /// <summary>
    /// Convert <see cref="Tensorflow.Tensor"/> to <see cref="TensorValue"/>.
    /// </summary>
    /// <param name="tensor">Tensorflow tensor.</param>
    /// <returns>Converted value.</returns>
    public static TensorValue ToValue(this Tensorflow.Tensor tensor)
    {
        return tensor.ToTensor();
    }

    /// <summary>
    /// Convert <see cref="Tensor"/> to <see cref="Tensorflow.Tensor"/>.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    /// <returns>Converted torch tensor.</returns>
    public static NDArray ToTFTensor(this Tensor tensor)
    {
        return new NDArray(tensor.BytesBuffer.ToArray(), tensor.Dimensions.ToArray(), tensor.ElementType.ToTFType());
    }

    private static readonly Dictionary<DataType, TF_DataType> _dataTypesToTorchType = new()
    {
        { DataType.Boolean, TF_DataType.TF_BOOL },
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
        { TF_DataType.TF_BOOL, DataType.Boolean },
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
