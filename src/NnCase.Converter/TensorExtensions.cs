using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;

namespace NnCase.Converter
{
    public static class TensorExtensions
    {
        public static Tensor<T> Transpose<T>(this Tensor<T> tensor, ReadOnlySpan<int> axes)
        {
            int inputExtSize = 4 - tensor.Rank;
            int outputExtSize = 4 - axes.Length;

            Span<int> extendedPerm = stackalloc int[4];
            for (int i = 0; i < outputExtSize; i++)
                extendedPerm[i] = i;
            for (int i = 0; i < axes.Length; i++)
                extendedPerm[i + outputExtSize] = axes[i] + inputExtSize;

            Span<int> outSizes = stackalloc int[4];
            Span<int> oldDims = stackalloc int[4] { 1, 1, 1, 1 };
            for (int i = 4 - tensor.Dimensions.Length; i < 4; i++)
                oldDims[i] = tensor.Dimensions[i - (4 - tensor.Dimensions.Length)];
            for (int i = 0; i < 4; i++)
                outSizes[i] = oldDims[extendedPerm[i]];

            Span<int> outp = stackalloc int[4];
            Span<int> inp = stackalloc int[4];
            var buffer = new T[tensor.Length];
            var destDimensions = new int[tensor.Dimensions.Length];
            for (int i = 0; i < axes.Length; i++)
                destDimensions[i] = tensor.Dimensions[axes[i]];
            var output = new DenseTensor<T>(buffer, outSizes);
            tensor = tensor.Reshape(oldDims);

            for (outp[3] = 0; outp[3] < outSizes[3]; outp[3]++)
            {
                inp[extendedPerm[3]] = outp[3];
                for (outp[2] = 0; outp[2] < outSizes[2]; outp[2]++)
                {
                    inp[extendedPerm[2]] = outp[2];
                    for (outp[1] = 0; outp[1] < outSizes[1]; outp[1]++)
                    {
                        inp[extendedPerm[1]] = outp[1];
                        for (outp[0] = 0; outp[0] < outSizes[0]; outp[0]++)
                        {
                            inp[extendedPerm[0]] = outp[0];
                            output[outp] = tensor[inp];
                        }
                    }
                }
            }

            return output.Reshape(destDimensions);
        }

        public static int GetSize(this ReadOnlySpan<int> shape)
        {
            int size = 1;
            for (int i = 0; i < shape.Length; i++)
                size *= shape[i];
            return size;
        }

        public static T[] ToArray<T>(this Tensor<T> tensor)
        {
            var dense = tensor.ToDenseTensor();
            return dense.Buffer.ToArray();
        }

        public static long Sum(this ReadOnlySpan<ushort> span)
        {
            long sum = 0;
            foreach (var item in span)
                sum += item;
            return sum;
        }
    }
}
