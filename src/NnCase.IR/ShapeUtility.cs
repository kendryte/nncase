using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR
{
    public static class ShapeUtility
    {
        public static int GetWindowedOutputSize(int input, int filter, int stride, int dilation, bool same)
        {
            var effectiveFilterSize = (filter - 1) * dilation + 1;
            if (same)
                return (input + stride - 1) / stride;
            else
                return (input - effectiveFilterSize + stride) / stride;
        }

        public static Shape GetTransposedShape(Shape inputShape, Shape perm)
        {
            var newShape = inputShape.Clone();
            for (int i = 0; i < newShape.Count; i++)
                newShape[i] = inputShape[perm[i]];
            return newShape;
        }

        public static Padding GetWindowedPadding(int input, int filter, int stride, int dilation, bool same)
        {
            return GetWindowedPadding(input, GetWindowedOutputSize(input, filter, stride, dilation, same), filter, stride, dilation);
        }

        public static Padding GetWindowedPadding(int input, int output, int filter, int stride, int dilation)
        {
            var effectiveFilterSize = (filter - 1) * dilation + 1;
            var padding = Math.Max(0, (output - 1) * stride + effectiveFilterSize - input);
            return new Padding { Before = padding / 2, After = padding - padding / 2 };
        }

        public static Shape NHWCToNCHW(Shape shape)
        {
            return GetTransposedShape(shape, new[] { 0, 3, 1, 2 });
        }

        public static Shape NCHWToNHWC(Shape shape)
        {
            return GetTransposedShape(shape, new[] { 0, 2, 3, 1 });
        }

        public static int GetBytes(DataType type)
        {
            switch (type)
            {
                case DataType.Float32:
                    return 4;
                case DataType.UInt8:
                    return 1;
                default:
                    throw new NotSupportedException($"Unsupported datatype: {type}");
            }
        }

        public static int ComputeSize(Shape shape)
        {
            int size = 1;
            foreach (var item in shape)
                size *= item;
            return size;
        }
    }
}
