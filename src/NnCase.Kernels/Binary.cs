using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void Binary(ReadOnlySpan<float> inputA, ReadOnlySpan<float> inputB, Span<float> output, in RuntimeShape inAShape, in RuntimeShape inBShape, in RuntimeShape outShape, ValueRange<float> fusedActivation, Func<float, float, float> binaryOp)
        {
            int outIdx = 0;
            for (int d0 = 0; d0 < outShape[0]; d0++)
            {
                for (int d1 = 0; d1 < outShape[1]; d1++)
                {
                    for (int d2 = 0; d2 < outShape[2]; d2++)
                    {
                        for (int d3 = 0; d3 < outShape[3]; d3++)
                        {
                            var inOff = new RuntimeShape(d0, d1, d2, d3);
                            var inAOff = GetReducedOffset(inOff, inAShape);
                            var inBOff = GetReducedOffset(inOff, inBShape);

                            var a = inputA[Offset(inAShape, inAOff)];
                            var b = inputB[Offset(inBShape, inBOff)];

                            output[outIdx++] = ApplyActivation(binaryOp(a, b), fusedActivation);
                        }
                    }
                }
            }
        }

        private static RuntimeShape GetReducedOffset(RuntimeShape inOff, RuntimeShape shape)
        {
            var off = new RuntimeShape();
            for (int i = 0; i < 4; i++)
            {
                if (inOff[i] >= shape[i])
                    off[i] = 0;
                else
                    off[i] = inOff[i];
            }

            return off;
        }
    }
}
