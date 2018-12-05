using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.Numerics.Tensors;
using System.Text;
using TensorFlow;

namespace NnCase.Converter
{
    public static class TensorflowExtensions
    {
        public static TFShape ToNHWC(this ReadOnlySpan<int> shape)
        {
            var dims = new long[shape.Length];

            // NC
            if (shape.Length == 1)
            {
                dims[0] = shape[0];
            }
            else if (shape.Length == 2)
            {
                dims[0] = shape[0];
                dims[1] = shape[1];
            }
            // NCHW
            else if (shape.Length == 4)
            {
                dims[0] = shape[0];
                dims[1] = shape[2];
                dims[2] = shape[3];
                dims[3] = shape[1];
            }
            else
            {
                throw new NotSupportedException();
            }

            return new TFShape(dims);
        }

        public static TFTensor ToNHWC<T>(this Tensor<T> tensor)
        {
            if (tensor.Dimensions.Length == 4)
                tensor = tensor.Transpose(new[] { 0, 2, 3, 1 });
            return tensor.ToTFTensor();
        }

        public static TFTensor ToHWIO<T>(this Tensor<T> tensor)
        {
            tensor = tensor.Transpose(new[] { 2, 3, 1, 0 });
            return tensor.ToTFTensor();
        }

        public static TFTensor ToTFTensor<T>(this Tensor<T> tensor)
        {
            var dims = tensor.Dimensions;
            var shapeArr = new long[dims.Length];
            for (int i = 0; i < dims.Length; i++)
                shapeArr[i] = dims[i];
            var shape = new TFShape(shapeArr);

            var count = tensor.Dimensions.GetSize();
            var buffer = (object)tensor.ToDenseTensor().Buffer.ToArray();

            if (typeof(T) == typeof(int))
                return TFTensor.FromBuffer(shape, (int[])buffer, 0, count);
            else if (typeof(T) == typeof(float))
                return TFTensor.FromBuffer(shape, (float[])buffer, 0, count);
            else
                throw new NotSupportedException();
        }

        public static TFOutput AddActivation(this TFGraph graph, TFOutput input, ActivationFunctionType activation)
        {
            switch (activation)
            {
                case ActivationFunctionType.Linear:
                    return input;
                case ActivationFunctionType.Relu:
                    return graph.Relu(input);
                case ActivationFunctionType.Relu6:
                    return graph.Relu6(input);
                default:
                    throw new NotSupportedException();
            }
        }
    }
}
