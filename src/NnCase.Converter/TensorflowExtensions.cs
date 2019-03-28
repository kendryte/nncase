using NnCase.Converter.Model.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
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

            if (shape.Length == 1)
            {
                dims[0] = shape[0];
            }
            // NC
            else if (shape.Length == 2)
            {
                dims[0] = -1;// shape[0];
                dims[1] = shape[1];
            }
            // NCHW
            else if (shape.Length == 4)
            {
                dims[0] = -1;// shape[0];
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

        public unsafe static Tensor<float> ToNCHW(this TFTensor tensor)
        {
            var span = new Span<float>(tensor.Data.ToPointer(), (int)tensor.TensorByteSize / 4);
            Tensor<float> dense = new DenseTensor<float>(span.ToArray(), tensor.Shape.Select(x => (int)x).ToArray());
            if (dense.Dimensions.Length == 4)
                dense = dense.Transpose(new[] { 0, 3, 1, 2 });
            return dense;
        }

        public static TFTensor ToHWIO<T>(this Tensor<T> tensor)
        {
            if (tensor.Dimensions.Length == 4)
                tensor = tensor.Transpose(new[] { 2, 3, 1, 0 });
            else if (tensor.Dimensions.Length == 2)
                tensor = tensor.Transpose(new[] { 1, 0 });
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

        public static int[] ToTFAxes(this ReadOnlySpan<int> axes)
        {
            if (axes.Length <= 2)
                return axes.ToArray();
            else
            {
                var newAxes = new int[axes.Length];
                newAxes[0] = axes[0];
                newAxes[1] = axes[axes.Length - 1];
                for (int i = 2; i < axes.Length - 1; i++)
                    newAxes[i] = axes[i];
                return newAxes;
            }
        }
    }
}
