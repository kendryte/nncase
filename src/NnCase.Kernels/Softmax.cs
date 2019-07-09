using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void Softmax(ReadOnlySpan<float> input, Span<float> output, float beta, int innerSize, int outerSize)
        {
            for (int batch = 0; batch < outerSize; batch++)
            {
                var src = input.Slice(batch * innerSize, innerSize);
                var dest = output.Slice(batch * innerSize, innerSize);

                var max = Max(src);
                float sum = 0;

                for (int i = 0; i < innerSize; i++)
                {
                    var value = MathF.Exp((src[i] - max) * beta);
                    sum += value;
                    dest[i] = value;
                }

                foreach (ref var d in dest)
                    d /= sum;
            }
        }

        private static float Max(ReadOnlySpan<float> values)
        {
            float max = float.MinValue;
            foreach (var v in values)
                max = Math.Max(max, v);
            return max;
        }
    }
}
