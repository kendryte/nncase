using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void Quantize(ReadOnlySpan<float> input, Span<byte> output, QuantizationParam param)
        {
            for (int i = 0; i < input.Length; i++)
            {
                var value = (int)Math.Round(input[i] * param.Scale + param.ZeroPoint);
                output[i] = (byte)Math.Clamp(value, 0, 255);
            }
        }

        public static void Dequantize(ReadOnlySpan<byte> input, Span<float> output, QuantizationParam param)
        {
            var div = 1.0f / param.Scale;

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (input[i] - param.ZeroPoint) * div;
            }
        }
    }
}
