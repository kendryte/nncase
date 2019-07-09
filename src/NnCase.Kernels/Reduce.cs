using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void Reduce(ReadOnlySpan<float> input, Span<float> output, in RuntimeShape inShape, in RuntimeShape outShape, float initialValue, Func<float, float, float> binaryOp)
        {
            output.Fill(initialValue);

            for (int d0 = 0; d0 < inShape[0]; d0++)
            {
                for (int d1 = 0; d1 < inShape[1]; d1++)
                {
                    for (int d2 = 0; d2 < inShape[2]; d2++)
                    {
                        for (int d3 = 0; d3 < inShape[3]; d3++)
                        {
                            var inOff = new RuntimeShape(d0, d1, d2, d3);
                            var outOff = GetReducedOffset(inOff, outShape);

                            var a = input[Offset(inShape, inOff)];
                            ref var b = ref output[Offset(outShape, outOff)];

                            b = binaryOp(a, b);
                        }
                    }
                }
            }
        }
    }
}
