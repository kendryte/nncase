using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void Unary(ReadOnlySpan<float> input, Span<float> output, Func<float, float> unaryOp)
        {
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = unaryOp(input[i]);
            }
        }
    }
}
