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

        public static Shape NormalizeReshape(Shape inputShape, Shape shape)
        {
            var newShape = shape.Clone();

            int shapeSize = 1;
            int nonDetId = -1;
            for (int i = 0; i < shape.Count; i++)
            {
                var v = shape[i];
                if (v == -1)
                {
                    if (nonDetId != -1)
                        throw new ArgumentException("Reshap can only have 1 non-determined dimension at most");
                    nonDetId = i;
                }
                else
                {
                    shapeSize *= v;
                }
            }

            if (nonDetId != -1)
                newShape[nonDetId] = ComputeSize(inputShape) / shapeSize;

            return newShape;
        }

        public static Shape GetBinaryOutputShape(Shape inputAShape, Shape inputBShape)
        {
            var outShape = new List<int>();

            var destDims = Math.Max(inputAShape.Count, inputBShape.Count);
            var inAExt = destDims - inputAShape.Count;
            var inBExt = destDims - inputBShape.Count;

            for (int i = 0; i < destDims; i++)
            {
                var inADim = i - inAExt;
                var inBDim = i - inBExt;

                var inA = inADim < 0 ? 1 : inputAShape[inADim];
                var inB = inBDim < 0 ? 1 : inputBShape[inBDim];
                if (inA == inB)
                    outShape.Add(inA);
                else if (inA == 1)
                    outShape.Add(inB);
                else if (inB == 1)
                    outShape.Add(inA);
                else
                    throw new ArgumentException("inputs are not compatible to broadcast");
            }

            return new Shape(outShape);
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
