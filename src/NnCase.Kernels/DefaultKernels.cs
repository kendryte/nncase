using System;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static int Offset(in RuntimeShape shape, in RuntimeShape index)
        {
            return ((index[0] * shape[1] + index[1]) * shape[2] + index[2]) * shape[3] + index[3];
        }

        public static int GetWindowedOutputSize(int input, int filter, int stride, int dilation, in Padding padding)
        {
            var effectiveFilterSize = (filter - 1) * dilation + 1;
            return (input + padding.Sum - effectiveFilterSize + stride) / stride;
        }

        public static float ApplyActivation(float value, in ValueRange<float> range)
        {
            return Math.Clamp(value, range.Min, range.Max);
        }

        public static Exception ThrowUnsupportedDataType()
        {
            return new NotSupportedException("Unsupported datatype");
        }
    }
}
