using System;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        private static int Offset(in RuntimeShape shape, in RuntimeShape index)
        {
            return ((index[0] * shape[1] + index[1]) * shape[2] + index[2]) * shape[3] + index[3];
        }

        private static int GetWindowedOutputSize(int input, int filter, int stride, int dilation, in Padding padding)
        {
            var effectiveFilterSize = (filter - 1) * dilation + 1;
            return (input + padding.Sum - effectiveFilterSize + stride) / stride;
        }

        private static float ApplyActivation(float value, in ValueRange<float> range)
        {
            return Math.Clamp(value, range.Min, range.Max);
        }

        private static Exception ThrowUnsupportedDataType()
        {
            return new NotSupportedException("Unsupported datatype");
        }
    }
}
