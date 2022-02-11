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

namespace Nncase.Evaluator
{
    public static class TorchExtentsion
    {
        /// <summary>
        /// convert torch tensor to const by gived shape.
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="ttype">target tensor type.</param>
        /// <returns></returns>
        /// <exception cref="InvalidCastException"></exception>
        public static Const ToConst(this torch.Tensor tensor, TensorType? ttype = null)
        {
            ttype ??= new TensorType(tensor.dtype.ToDataType(), new Shape(tensor.shape));
            if (ttype.Shape.Prod().FixedValue != tensor.numel())
            {
                throw new InvalidCastException($"The Target Shape Prod != {tensor.numel()}!");
            }

            if (!tensor.is_contiguous())
                tensor = tensor.contiguous();
            return new Const(ttype, tensor.bytes.ToArray());
        }

        /// <summary>
        /// wrapper for python use.
        /// </summary>
        /// <param name="tensor"></param>
        /// <returns></returns>
        public static byte[] ToSpan(this torch.Tensor tensor) => tensor.bytes.ToArray();

        /// <summary>
        /// convert const to torch tensor
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public static torch.Tensor ToTorchTensor(this Const expr)
        {
            var dtype = expr.ValueType.DType;
            var shape = expr.ValueType.IsScalar
                ? new long[] { }
                : expr.ValueType.Shape.ToList().Select(x => (long)x.FixedValue).ToArray();
            return dtype switch
            {
                PrimType ptype => ptype switch
                {
                    { TypeCode: PrimTypeCode.Int8, Lanes: 1 } => torch.tensor(expr.ToTensor<sbyte>(), shape, ToTorchType(dtype)),
                    { TypeCode: PrimTypeCode.Int16, Lanes: 1 } => torch.tensor(expr.ToTensor<short>(), shape, ToTorchType(dtype)),
                    { TypeCode: PrimTypeCode.Int32, Lanes: 1 } => torch.tensor(expr.ToTensor<int>(), shape, ToTorchType(dtype)),
                    { TypeCode: PrimTypeCode.Int64, Lanes: 1 } => torch.tensor(expr.ToTensor<long>(), shape, ToTorchType(dtype)),
                    { TypeCode: PrimTypeCode.UInt8, Lanes: 1 } => torch.tensor(expr.ToTensor<byte>(), shape, ToTorchType(dtype)),
                    // DataType.UInt16 => torch.tensor(expr.ToTensor<ushort>(), shape, ToTorchType(dtype)),
                    // DataType.UInt32 => torch.tensor(expr.ToTensor<uint>(), shape, ToTorchType(dtype)),
                    // DataType.UInt64 => torch.tensor(expr.ToTensor<ulong>(), shape, ToTorchType(dtype)),
                    // DataType.Float16 => torch.tensor(expr.ToTensor<Float16>(), shape, ToTorchType(dtype)),
                    { TypeCode: PrimTypeCode.Float32, Lanes: 1 } => torch.tensor(expr.ToTensor<float>(), shape, ToTorchType(dtype)),
                    { TypeCode: PrimTypeCode.Float64, Lanes: 1 } => torch.tensor(expr.ToTensor<double>(), shape, ToTorchType(dtype)),
                    // {PrimTypeCode:PrimTypeCode.BFloat16,Lanes:1} => torch.tensor(expr.ToTensor<BFloat16>(), shape, ToTorchType(dtype)),
                    { TypeCode: PrimTypeCode.Bool, Lanes: 1 } => torch.tensor(expr.ToTensor<bool>(), shape, ToTorchType(dtype)),
                    // DataType.String => torch.tensor(expr.ToTensor<>(), shape, ToTorchType(dtype)),
                    _ => throw new ArgumentOutOfRangeException("Unsupported conversion for datatype to torch.ScalarType")
                },
                _ => throw new ArgumentOutOfRangeException($"Unsupported conversion for {dtype.GetType().Name}")
            };
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

        public static torch.ScalarType ToTorchType(this DataType dt) => _dataTypesToTorchType[dt];

        public static DataType ToDataType(this torch.ScalarType dt) => _TorchTypeTodataTypes[dt];

        public static PaddingModes ToTorch(this PadMode mode) => mode switch
        {
            PadMode.Constant => PaddingModes.Constant,
            PadMode.Reflect => PaddingModes.Reflect,
            _ => throw new NotImplementedException($"The Pytorch Can't Accept {mode.ToString()} Mode!"),
        };
    }
}
