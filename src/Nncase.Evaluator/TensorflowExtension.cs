using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Tensorflow;
using Shape = Tensorflow.Shape;

namespace Nncase.Evaluator
{
    public static class TensorflowExtension
    {
        public static Const ToConst(this Tensor tensor)
        {
            return new Const(
                new TensorType(ToDataType(tensor.dtype), tensor.shape.as_int_list()),
                // todo:this is copy
                tensor.BufferToArray());
        }

        public static Tensor ToTFTensor(this Const tensor)
        {
            var shape = new Shape(tensor.CheckedShape.ToValueArray());
            return new Tensor(tensor.Data.Data(), shape, tensor.CheckedDataType.ToTFType());
        }

        private static readonly Dictionary<DataType, TF_DataType> _dataTypesToTorchType = new()
        {
            {DataType.Bool, TF_DataType.DtBoolRef},
            {DataType.Int8, TF_DataType.DtInt8Ref},
            {DataType.Int16, TF_DataType.DtInt16Ref},
            {DataType.Int32, TF_DataType.DtInt32Ref},
            {DataType.Int64, TF_DataType.DtInt64Ref},
            {DataType.UInt8, TF_DataType.DtUint8Ref},
            {DataType.Float16, TF_DataType.DtHalfRef},
            {DataType.Float32, TF_DataType.DtFloatRef},
            {DataType.Float64, TF_DataType.DtDoubleRef},
        };

        private static readonly Dictionary<TF_DataType, DataType> _TorchTypeTodataTypes = new()
        {
            {TF_DataType.DtBoolRef, DataType.Bool},
            {TF_DataType.DtInt8Ref, DataType.Int8},
            {TF_DataType.DtInt16Ref, DataType.Int16},
            {TF_DataType.DtInt32Ref, DataType.Int32},
            {TF_DataType.DtInt64Ref, DataType.Int64},
            {TF_DataType.DtUint8Ref, DataType.UInt8},
            {TF_DataType.DtHalfRef, DataType.Float16},
            {TF_DataType.DtFloatRef, DataType.Float32},
            {TF_DataType.DtDoubleRef, DataType.Float64},
        };
        
        public static TF_DataType ToTFType(this DataType dt) => _dataTypesToTorchType[dt];

        public static DataType ToDataType(this TF_DataType dt) => _TorchTypeTodataTypes[dt];
    }
}