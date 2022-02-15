// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using TorchSharp;

namespace Nncase.Evaluator;

/// <summary>
/// Torch extensions.
/// </summary>
public static class TorchExtentsion
{
    /// <summary>
    /// Convert <see cref="torch.Tensor"/> to <see cref="Tensor"/>.
    /// </summary>
    /// <param name="tensor">Torch tensor.</param>
    /// <returns>Converted tensor.</returns>
    public static Tensor ToTensor(this torch.Tensor tensor)
    {
        if (!tensor.is_contiguous())
        {
            tensor = tensor.contiguous();
        }

        return tensor.ToTensor(new Shape(tensor.shape));
    }

    /// <summary>
    /// Convert <see cref="torch.Tensor"/> to <see cref="Tensor"/>.
    /// </summary>
    /// <param name="tensor">Torch tensor.</param>
    /// <param name="dimensions">Dimensions.</param>
    /// <returns>Converted tensor.</returns>
    public static Tensor ToTensor(this torch.Tensor tensor, ReadOnlySpan<int> dimensions)
    {
        if (TensorUtilities.GetProduct(dimensions) != tensor.numel())
        {
            throw new InvalidCastException($"The Target Shape Prod {TensorUtilities.GetProduct(dimensions)} != {tensor.numel()}!");
        }

        if (!tensor.is_contiguous())
        {
            tensor = tensor.contiguous();
        }

        // TODO: Copy-free
        return Tensor.FromBytes(ToDataType(tensor.dtype), tensor.bytes, dimensions);
    }

    /// <summary>
    /// Convert <see cref="torch.Tensor"/> to <see cref="TensorValue"/>.
    /// </summary>
    /// <param name="tensor">Torch tensor.</param>
    /// <returns>Converted value.</returns>
    public static TensorValue ToValue(this torch.Tensor tensor)
    {
        return tensor.ToTensor();
    }

    /// <summary>
    /// Convert <see cref="Tensor"/> to <see cref="torch.Tensor"/>.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    /// <returns>Converted torch tensor.</returns>
    public static torch.Tensor ToTorchTensor(this Tensor tensor)
    {
        // torch.as_tensor()
        var dtype = tensor.ElementType;
        var shape = tensor.Dimensions.AsValueEnumerable().Select(x => (long)x).ToArray();
        return dtype switch
        {
            { ElemType: ElemType.Int8, Lanes: 1 } => torch.tensor(tensor.Cast<sbyte>(), shape, ToTorchType(dtype)),
            { ElemType: ElemType.Int16, Lanes: 1 } => torch.tensor(tensor.Cast<short>(), shape, ToTorchType(dtype)),
            { ElemType: ElemType.Int32, Lanes: 1 } => torch.tensor(tensor.Cast<int>(), shape, ToTorchType(dtype)),
            { ElemType: ElemType.Int64, Lanes: 1 } => torch.tensor(tensor.Cast<long>(), shape, ToTorchType(dtype)),
            { ElemType: ElemType.UInt8, Lanes: 1 } => torch.tensor(tensor.Cast<byte>(), shape, ToTorchType(dtype)),

            // DataType.UInt16 => torch.tensor(expr.ToTensor<ushort>(), shape, ToTorchType(dtype)),
            // DataType.UInt32 => torch.tensor(expr.ToTensor<uint>(), shape, ToTorchType(dtype)),
            // DataType.UInt64 => torch.tensor(expr.ToTensor<ulong>(), shape, ToTorchType(dtype)),
            // DataType.Float16 => torch.tensor(expr.ToTensor<Float16>(), shape, ToTorchType(dtype)),
            { ElemType: ElemType.Float32, Lanes: 1 } => torch.tensor(tensor.Cast<float>(), shape, ToTorchType(dtype)),
            { ElemType: ElemType.Float64, Lanes: 1 } => torch.tensor(tensor.Cast<double>(), shape, ToTorchType(dtype)),

            // {ElemType:ElemType.BFloat16,Lanes:1} => torch.tensor(expr.ToTensor<BFloat16>(), shape, ToTorchType(dtype)),
            { ElemType: ElemType.Bool, Lanes: 1 } => torch.tensor(tensor.Cast<bool>(), shape, ToTorchType(dtype)),

            // DataType.String => torch.tensor(expr.ToTensor<>(), shape, ToTorchType(dtype)),
            _ => throw new ArgumentOutOfRangeException("Unsupported conversion for datatype to torch.ScalarType"),
        };
    }

    private static readonly Dictionary<DataType, torch.ScalarType> _dataTypesToTorchType = new()
    {
        { DataType.Boolean, torch.ScalarType.Bool },
        { DataType.Int8, torch.ScalarType.Int8 },
        { DataType.Int16, torch.ScalarType.Int16 },
        { DataType.Int32, torch.ScalarType.Int32 },
        { DataType.Int64, torch.ScalarType.Int64 },
        { DataType.UInt8, torch.ScalarType.Byte },
        { DataType.Float16, torch.ScalarType.Float16 },
        { DataType.Float32, torch.ScalarType.Float32 },
        { DataType.Float64, torch.ScalarType.Float64 },
    };
    private static readonly Dictionary<torch.ScalarType, DataType> _TorchTypeTodataTypes = new()
    {
        { torch.ScalarType.Bool, DataType.Boolean },
        { torch.ScalarType.Int8, DataType.Int8 },
        { torch.ScalarType.Int16, DataType.Int16 },
        { torch.ScalarType.Int32, DataType.Int32 },
        { torch.ScalarType.Int64, DataType.Int64 },
        { torch.ScalarType.Byte, DataType.UInt8 },
        { torch.ScalarType.Float16, DataType.Float16 },
        { torch.ScalarType.Float32, DataType.Float32 },
        { torch.ScalarType.Float64, DataType.Float64 },
    };

    public static torch.ScalarType ToTorchType(this DataType dt) => _dataTypesToTorchType[dt];

    public static DataType ToDataType(this torch.ScalarType dt) => _TorchTypeTodataTypes[dt];

    public static PaddingModes ToTorch(this PadMode mode) => mode switch
    {
        PadMode.Constant => PaddingModes.Constant,
        PadMode.Reflect => PaddingModes.Reflect,
        _ => throw new NotImplementedException($"The Pytorch Can't Accept {mode.ToString()} Mode!"),
    };
}
