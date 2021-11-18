using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using TorchSharp;

namespace Nncase.Evaluator
{
    public static class TorchExtentsion
    {
        public static Const ToConst(this torch.Tensor tensor) => 
            new Const(new TensorType(ToDataType(tensor.dtype), new Shape(tensor.shape)), 
                tensor.bytes.ToArray());
        public static torch.Tensor ToTorchTensor(this Const expr)
        {
            // null checked type
            var dtype = (expr.CheckedType as TensorType).DType;
            if (expr.ValueType.IsScalar)
            {
                return dtype switch
                {
                    DataType.Int8 => torch.tensor(expr.ToScalar<sbyte>(), ToTorchDataType(dtype)),
                    DataType.Int16 => torch.tensor(expr.ToScalar<short>(), ToTorchDataType(dtype)),
                    DataType.Int32 => torch.tensor(expr.ToScalar<int>(), ToTorchDataType(dtype)),
                    DataType.Int64 => torch.tensor(expr.ToScalar<long>(), ToTorchDataType(dtype)),
                    DataType.UInt8 => torch.tensor(expr.ToScalar<byte>(), ToTorchDataType(dtype)),
                    // DataType.UInt16 => torch.tensor(expr.ToScalar<ushort>(), ToTorchDataType(dtype)),
                    // DataType.UInt32 => torch.tensor(expr.ToScalar<uint>(), ToTorchDataType(dtype)),
                    // DataType.UInt64 => torch.tensor(expr.ToScalar<ulong>(), ToTorchDataType(dtype)),
                    // DataType.Float16 => torch.tensor(expr.ToScalar<Float16>(), ToTorchDataType(dtype)),
                    DataType.Float32 => torch.tensor(expr.ToScalar<float>(), ToTorchDataType(dtype)),
                    DataType.Float64 => torch.tensor(expr.ToScalar<double>(), ToTorchDataType(dtype)),
                    // DataType.BFloat16 => torch.tensor(expr.ToScalar<BFloat16>(), ToTorchDataType(dtype)),
                    DataType.Bool => torch.tensor(expr.ToScalar<bool>(), ToTorchDataType(dtype)),
                    // DataType.String => torch.tensor(expr.ToScalar<>(), ToTorchDataType(dtype)),
                    _ => throw new ArgumentOutOfRangeException("Unsupported conversion for datatype to torch.ScalarType")
                };
            }
            else
            {
                var shape = expr.CheckedShape.ToList().Select(x => (long)x.FixedValue).ToArray();
                return dtype switch
                {
                    DataType.Int8 => torch.tensor(expr.ToTensor<sbyte>(), shape, ToTorchDataType(dtype)),
                    DataType.Int16 => torch.tensor(expr.ToTensor<short>(), shape, ToTorchDataType(dtype)),
                    DataType.Int32 => torch.tensor(expr.ToTensor<int>(), shape, ToTorchDataType(dtype)),
                    DataType.Int64 => torch.tensor(expr.ToTensor<long>(), shape, ToTorchDataType(dtype)),
                    DataType.UInt8 => torch.tensor(expr.ToTensor<byte>(), shape, ToTorchDataType(dtype)),
                    // DataType.UInt16 => torch.tensor(expr.ToTensor<ushort>(), shape, ToTorchDataType(dtype)),
                    // DataType.UInt32 => torch.tensor(expr.ToTensor<uint>(), shape, ToTorchDataType(dtype)),
                    // DataType.UInt64 => torch.tensor(expr.ToTensor<ulong>(), shape, ToTorchDataType(dtype)),
                    // DataType.Float16 => torch.tensor(expr.ToTensor<Float16>(), shape, ToTorchDataType(dtype)),
                    DataType.Float32 => torch.tensor(expr.ToTensor<float>(), shape, ToTorchDataType(dtype)),
                    DataType.Float64 => torch.tensor(expr.ToTensor<double>(), shape, ToTorchDataType(dtype)),
                    // DataType.BFloat16 => torch.tensor(expr.ToTensor<BFloat16>(), shape, ToTorchDataType(dtype)),
                    DataType.Bool => torch.tensor(expr.ToTensor<bool>(), shape, ToTorchDataType(dtype)),
                    // DataType.String => torch.tensor(expr.ToTensor<>(), shape, ToTorchDataType(dtype)),
                    _ => throw new ArgumentOutOfRangeException("Unsupported conversion for datatype to torch.ScalarType")
                };
            }
        }

        private static readonly Dictionary<DataType, torch.ScalarType> _dataTypesToTorchType = new()
        {
            { DataType.UInt8, torch.ScalarType.Byte },
            { DataType.Int32, torch.ScalarType.Int32 },
            { DataType.Float32, torch.ScalarType.Float32 },
            { DataType.Float64, torch.ScalarType.Float64 },
            { DataType.Bool, torch.ScalarType.Bool },
        };
        private static readonly Dictionary<torch.ScalarType, DataType> _TorchTypeTodataTypes = new()
        {
            { torch.ScalarType.Byte, DataType.UInt8 },
            { torch.ScalarType.Int32, DataType.Int32 },
            { torch.ScalarType.Float32, DataType.Float32 },
            { torch.ScalarType.Float64, DataType.Float64 },
            { torch.ScalarType.Bool, DataType.Bool },
        };

        public static torch.ScalarType ToTorchDataType(DataType dt) => _dataTypesToTorchType[dt];


        public static DataType ToDataType(torch.ScalarType dt) => _TorchTypeTodataTypes[dt];
    }
}
