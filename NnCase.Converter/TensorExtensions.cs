using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter
{
    static class TensorExtensions
    {
        public static Tensor<T> Transpose<T>(this Tensor<T> tensor, ReadOnlySpan<int> axes)
        {
            var sourceStrides = tensor.Strides.ToArray();
            var destStrides = new int[sourceStrides.Length];
            for (int i = 0; i < axes.Length; i++)
                destStrides[i] = sourceStrides[axes[i]];

            var buffer = new T[tensor.Length];
            for (int i = 0; i < buffer.Length; i++)
            {
                var idx = TransformIndexByStrides(i, sourceStrides, false, destStrides);
                buffer[idx] = tensor.GetValue(i);
            }

            var destDimensions = new int[sourceStrides.Length];
            for (int i = 0; i < axes.Length; i++)
                destDimensions[i] = tensor.Dimensions[axes[i]];
            return new DenseTensor<T>(buffer, destDimensions);
        }

        private static int TransformIndexByStrides(int index, int[] sourceStrides, bool sourceReverseStride, int[] transformStrides)
        {
            int transformIndex = 0;
            int remainder = index;

            for (int i = 0; i < sourceStrides.Length; i++)
            {
                // reverse the index for reverseStride so that we divide by largest stride first
                var nIndex = sourceReverseStride ? sourceStrides.Length - 1 - i : i;

                var sourceStride = sourceStrides[nIndex];
                var transformStride = transformStrides[nIndex];

                transformIndex += transformStride * (remainder / sourceStride);
                remainder %= sourceStride;
            }

            return transformIndex;
        }
    }
}
