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
    /// convert torch tensor to Tensor by gived shape.
    /// </summary>
    /// <param name="tensor"></param>
    /// <param name="ttype">target tensor type.</param>
    /// <returns></returns>
    /// <exception cref="InvalidCastException"></exception>
    public static Tensor ToTensor(this torch.Tensor tensor, TensorType? ttype = null)
    {
        ttype ??= new TensorType(tensor.dtype.ToDataType(), new Shape(tensor.shape));
        if (ttype.Shape.Prod().FixedValue != tensor.numel())
        {
            throw new InvalidCastException($"The Target Shape Prod != {tensor.numel()}!");
        }
        if (!tensor.is_contiguous())
            tensor = tensor.contiguous();
        return Tensor.FromBytes(ToDataType(tensor.dtype), tensor.bytes, new Shape(tensor.shape));
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
        var shape = tensor.Dimensions.AsValueEnumerable().Select(x => (long)x).ToArray();
        if (tensor.ElementType is PrimType dtype)
        {
            return dtype switch
            {
                { TypeCode: PrimTypeCode.Int8, Lanes: 1 } => torch.tensor(tensor.Cast<sbyte>(), shape, ToTorchType(dtype)),
                { TypeCode: PrimTypeCode.Int16, Lanes: 1 } => torch.tensor(tensor.Cast<short>(), shape, ToTorchType(dtype)),
                { TypeCode: PrimTypeCode.Int32, Lanes: 1 } => torch.tensor(tensor.Cast<int>(), shape, ToTorchType(dtype)),
                { TypeCode: PrimTypeCode.Int64, Lanes: 1 } => torch.tensor(tensor.Cast<long>(), shape, ToTorchType(dtype)),
                { TypeCode: PrimTypeCode.UInt8, Lanes: 1 } => torch.tensor(tensor.Cast<byte>(), shape, ToTorchType(dtype)),
                { TypeCode: PrimTypeCode.Float32, Lanes: 1 } => torch.tensor(tensor.Cast<float>(), shape, ToTorchType(dtype)),
                { TypeCode: PrimTypeCode.Float64, Lanes: 1 } => torch.tensor(tensor.Cast<double>(), shape, ToTorchType(dtype)),
                { TypeCode: PrimTypeCode.Bool, Lanes: 1 } => torch.tensor(tensor.Cast<bool>(), shape, ToTorchType(dtype)),
                _ => throw new ArgumentOutOfRangeException("Unsupported conversion for datatype to torch.ScalarType"),
            };
        }
        else if (tensor.ElementType is PointerType { ElemType: PrimType { } })
        {
            return torch.tensor(tensor.Cast<long>(), shape, torch.ScalarType.Int64);
        }
        throw new NotSupportedException($"Can't Convert TensorType {tensor.ElementType.ToString()} to TorchTensor");
    }

    private static readonly Dictionary<DataType, torch.ScalarType> _dataTypesToTorchType = new()
    {
        { DataType.Bool, torch.ScalarType.Bool },
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
        { torch.ScalarType.Bool, DataType.Bool },
        { torch.ScalarType.Int8, DataType.Int8 },
        { torch.ScalarType.Int16, DataType.Int16 },
        { torch.ScalarType.Int32, DataType.Int32 },
        { torch.ScalarType.Int64, DataType.Int64 },
        { torch.ScalarType.Byte, DataType.UInt8 },
        { torch.ScalarType.Float16, DataType.Float16 },
        { torch.ScalarType.Float32, DataType.Float32 },
        { torch.ScalarType.Float64, DataType.Float64 },
    };

    /// <summary>
    /// convert the datatype to torch type
    /// </summary>
    /// <param name="dt"></param>
    /// <returns></returns>
    public static torch.ScalarType ToTorchType(this DataType dt) => _dataTypesToTorchType[dt];

    /// <summary>
    /// convert torch type to datatype
    /// </summary>
    /// <param name="dt"></param>
    /// <returns></returns>
    public static DataType ToDataType(this torch.ScalarType dt) => _TorchTypeTodataTypes[dt];

    /// <summary>
    /// convert the pad mode
    /// </summary>
    /// <param name="mode"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public static PaddingModes ToTorch(this PadMode mode) => mode switch
    {
        PadMode.Constant => PaddingModes.Constant,
        PadMode.Reflect => PaddingModes.Reflect,
        _ => throw new NotImplementedException($"The Pytorch Can't Accept {mode.ToString()} Mode!"),
    };
}
