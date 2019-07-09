using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void MatMul(ReadOnlySpan<float> inputA, ReadOnlySpan<float> inputB, Span<float> output, ReadOnlySpan<float> bias, int aRows, int aCols, int bCols, ValueRange<float> fusedActivation)
        {
            for (int oy = 0; oy < aRows; oy++)
            {
                for (int ox = 0; ox < bCols; ox++)
                {
                    float value = bias[ox];
                    for (int i = 0; i < aCols; i++)
                    {
                        var a = inputA[oy * aCols + i];
                        var b = inputB[i * bCols + ox];
                        value += a * b;
                    }

                    output[oy * bCols + ox] = ApplyActivation(value, fusedActivation);
                }
            }
        }
    }
}
