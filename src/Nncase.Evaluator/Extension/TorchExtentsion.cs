// // Copyright (c) Canaan Inc. All rights reserved.
// // Licensed under the Apache license. See LICENSE file in the project root for full license information.
//
// using System;
// using System.Collections.Generic;
// using System.IO;
// using System.Linq;
// using System.Text;
// using System.Threading.Tasks;
// using NetFabric.Hyperlinq;
// using Nncase.IR;
// using TorchSharp;
//
// namespace Nncase.Evaluator;
//
// /// <summary>
// /// Torch extensions.
// /// </summary>
// public static class TorchExtentsion
// {
//     /// <summary>
//     /// Convert <see cref="torch.Tensor"/> to <see cref="Tensor"/>.
//     /// </summary>
//     /// <param name="tensor">Torch tensor.</param>
//     /// <returns>Converted tensor.</returns>
//     public static Tensor ToTensor(this torch.Tensor tensor)
//     {
//         if (!tensor.is_contiguous())
//         {
//             tensor = tensor.contiguous();
//         }
//
//         return tensor.ToTensor(new Shape(tensor.shape));
//     }
//
//     /// <summary>
//     /// Convert <see cref="torch.Tensor"/> to <see cref="Tensor"/>.
//     /// </summary>
//     /// <param name="tensor">Torch tensor.</param>
//     /// <param name="dimensions">Dimensions.</param>
//     /// <returns>Converted tensor.</returns>
//     public static Tensor ToTensor(this torch.Tensor tensor, ReadOnlySpan<int> dimensions)
//     {
//         if (TensorUtilities.GetProduct(dimensions) != tensor.numel())
//         {
//             throw new InvalidCastException($"The Target Shape Prod {TensorUtilities.GetProduct(dimensions)} != {tensor.numel()}!");
//         }
//
//         if (!tensor.is_contiguous())
//         {
//             tensor = tensor.contiguous();
//         }
//
//         // TODO: Copy-free
//         return Tensor.FromBytes(ToDataType(tensor.dtype), tensor.bytes, dimensions);
//     }
//
//     /// <summary>
//     /// Convert <see cref="torch.Tensor"/> to <see cref="TensorValue"/>.
//     /// </summary>
//     /// <param name="tensor">Torch tensor.</param>
//     /// <returns>Converted value.</returns>
//     public static TensorValue ToValue(this torch.Tensor tensor)
//     {
//         return tensor.ToTensor();
//     }
//
//     /// <summary>
//     /// Convert <see cref="Tensor"/> to <see cref="torch.Tensor"/>.
//     /// </summary>
//     /// <param name="tensor">Tensor.</param>
//     /// <returns>Converted torch tensor.</returns>
//     public static torch.Tensor ToTorchTensor(this Tensor tensor)
//     {
//         // torch.as_tensor()
//         var dtype = tensor.ElementType;
//         var shape = tensor.Dimensions.AsValueEnumerable().Select(x => (long)x).ToArray();
//         return _converters[dtype](tensor, shape);
//     }
//
//     private static readonly Dictionary<DataType, Func<Tensor, long[], torch.Tensor>> _converters = new()
//     {
//         { DataTypes.Int8, (tensor, shape) => torch.tensor(tensor.Cast<sbyte>(), shape, torch.ScalarType.Int8) },
//         { DataTypes.Int16, (tensor, shape) => torch.tensor(tensor.Cast<short>(), shape, torch.ScalarType.Int16) },
//         { DataTypes.Int32, (tensor, shape) => torch.tensor(tensor.Cast<int>(), shape, torch.ScalarType.Int32) },
//         { DataTypes.Int64, (tensor, shape) => torch.tensor(tensor.Cast<long>(), shape, torch.ScalarType.Int64) },
//         { DataTypes.UInt8, (tensor, shape) => torch.tensor(tensor.Cast<byte>(), shape, torch.ScalarType.Byte) },
//         { DataTypes.Float32, (tensor, shape) => torch.tensor(tensor.Cast<float>(), shape, torch.ScalarType.Float32) },
//         { DataTypes.Float64, (tensor, shape) => torch.tensor(tensor.Cast<float>(), shape, torch.ScalarType.Float64) },
//     };
//
//     private static readonly Dictionary<DataType, torch.ScalarType> _dataTypesToTorchType = new()
//     {
//         { DataTypes.Boolean, torch.ScalarType.Bool },
//         { DataTypes.Int8, torch.ScalarType.Int8 },
//         { DataTypes.Int16, torch.ScalarType.Int16 },
//         { DataTypes.Int32, torch.ScalarType.Int32 },
//         { DataTypes.Int64, torch.ScalarType.Int64 },
//         { DataTypes.UInt8, torch.ScalarType.Byte },
//         { DataTypes.Float16, torch.ScalarType.Float16 },
//         { DataTypes.Float32, torch.ScalarType.Float32 },
//         { DataTypes.Float64, torch.ScalarType.Float64 },
//     };
//     private static readonly Dictionary<torch.ScalarType, DataType> _TorchTypeTodataTypes = new()
//     {
//         { torch.ScalarType.Bool, DataTypes.Boolean },
//         { torch.ScalarType.Int8, DataTypes.Int8 },
//         { torch.ScalarType.Int16, DataTypes.Int16 },
//         { torch.ScalarType.Int32, DataTypes.Int32 },
//         { torch.ScalarType.Int64, DataTypes.Int64 },
//         { torch.ScalarType.Byte, DataTypes.UInt8 },
//         { torch.ScalarType.Float16, DataTypes.Float16 },
//         { torch.ScalarType.Float32, DataTypes.Float32 },
//         { torch.ScalarType.Float64, DataTypes.Float64 },
//     };
//
//     public static torch.ScalarType ToTorchType(this DataType dt) => _dataTypesToTorchType[dt];
//
//     public static DataType ToDataType(this torch.ScalarType dt) => _TorchTypeTodataTypes[dt];
//
//     public static PaddingModes ToTorch(this PadMode mode) => mode switch
//     {
//         PadMode.Constant => PaddingModes.Constant,
//         PadMode.Reflect => PaddingModes.Reflect,
//         _ => throw new NotImplementedException($"The Pytorch Can't Accept {mode.ToString()} Mode!"),
//     };
// }
