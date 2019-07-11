using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Kernels;
using NnCase.Targets.K210.IR;

namespace NnCase.Targets.K210.Kernels
{
    public static partial class K210Kernels
    {
        public static void DepthwiseConv2D(Span<float> input, Span<float> output, Span<float> weights, Span<float> bias, in RuntimeShape inShape, KPUFilterType filterType, ValueRange<float> fusedActivation)
        {
            if (filterType == KPUFilterType.Filter_1x1)
                DepthwiseConv2D_1x1(input, output, weights, bias, inShape, fusedActivation);
            else
                DepthwiseConv2D_3x3(input, output, weights, bias, inShape, fusedActivation);
        }

        private static void DepthwiseConv2D_1x1(Span<float> input, Span<float> output, Span<float> weights, Span<float> bias, in RuntimeShape inShape, ValueRange<float> fusedActivation)
        {
            int outputIdx = 0;

            for (int batch = 0; batch < inShape[0]; batch++)
            {
                var inBatch = input.Slice(batch * inShape[1] * inShape[2] * inShape[3], inShape[1] * inShape[2] * inShape[3]);

                for (int oc = 0; oc < inShape[1]; oc++)
                {
                    float w = weights[oc];

                    for (int oy = 0; oy < inShape[2]; oy++)
                    {
                        for (int ox = 0; ox < inShape[3]; ox++)
                        {
                            float value = bias[oc];
                            float inV = inBatch[(oc * inShape[2] + oy) * inShape[3] + ox];

                            value += inV * w;

                            output[outputIdx++] = DefaultKernels.ApplyActivation(value, fusedActivation);
                        }
                    }
                }
            }
        }

        private static void DepthwiseConv2D_3x3(Span<float> input, Span<float> output, Span<float> weights, Span<float> bias, in RuntimeShape inShape, ValueRange<float> fusedActivation)
        {
            int outputIdx = 0;

            for (int batch = 0; batch < inShape[0]; batch++)
            {
                var inBatch = input.Slice(batch * inShape[1] * inShape[2] * inShape[3], inShape[1] * inShape[2] * inShape[3]);

                for (int oc = 0; oc < inShape[1]; oc++)
                {
                    var wOC = weights.Slice(oc * 3 * 3, 3 * 3);

                    for (int oy = 0; oy < inShape[2]; oy++)
                    {
                        for (int ox = 0; ox < inShape[3]; ox++)
                        {
                            int inYOrigin = oy - 1;
                            int inXOrigin = ox - 1;
                            int filterYStart = Math.Max(0, -inYOrigin);
                            int filterYEnd = Math.Min(3, inShape[2] - inYOrigin);
                            int filterXSstart = Math.Max(0, -inXOrigin);
                            int filterXEnd = Math.Min(3, inShape[3] - inXOrigin);
                            float value = bias[oc];

                            for (int ky = filterYStart; ky < filterYEnd; ky++)
                            {
                                for (int kx = filterXSstart; kx < filterXEnd; kx++)
                                {
                                    int inY = inYOrigin + ky;
                                    int inX = inXOrigin + kx;

                                    float inV = inBatch[(oc * inShape[2] + inY) * inShape[3] + inX];
                                    float w = wOC[ky * 3 + kx];

                                    value += inV * w;
                                }
                            }

                            output[outputIdx++] = DefaultKernels.ApplyActivation(value, fusedActivation);
                        }
                    }
                }
            }
        }
    }
}
