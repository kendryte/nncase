using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Kernels;
using NnCase.Targets.K210.IR;

namespace NnCase.Targets.K210.Kernels
{
    public static partial class K210Kernels
    {
        public static void Conv2D(Span<float> input, Span<float> output, Span<float> weights, Span<float> bias, in RuntimeShape inShape, int outputChannels, KPUFilterType filterType, ValueRange<float> fusedActivation)
        {
            if (filterType == KPUFilterType.Filter_1x1)
                Conv2D_1x1(input, output, weights, bias, inShape, outputChannels, fusedActivation);
            else
                Conv2D_3x3(input, output, weights, bias, inShape, outputChannels, fusedActivation);
        }

        private static void Conv2D_1x1(Span<float> input, Span<float> output, Span<float> weights, Span<float> bias, in RuntimeShape inShape, int outputChannels, ValueRange<float> fusedActivation)
        {
            var filterShape = new RuntimeShape(outputChannels, inShape[1], 1, 1);
            int outputIdx = 0;

            for (int batch = 0; batch < inShape[0]; batch++)
            {
                var inBatch = input.Slice(batch * inShape[1] * inShape[2] * inShape[3], inShape[1] * inShape[2] * inShape[3]);

                for (int oc = 0; oc < outputChannels; oc++)
                {
                    var wOC = weights.Slice(oc * filterShape[1], filterShape[1]);

                    for (int oy = 0; oy < inShape[2]; oy++)
                    {
                        for (int ox = 0; ox < inShape[3]; ox++)
                        {
                            float value = bias[oc];

                            for (int ic = 0; ic < inShape[1]; ic++)
                            {
                                float inV = inBatch[(ic * inShape[2] + oy) * inShape[3] + ox];
                                float w = wOC[ic];

                                value += inV * w;
                            }

                            output[outputIdx++] = DefaultKernels.ApplyActivation(value, fusedActivation);
                        }
                    }
                }
            }
        }

        private static void Conv2D_3x3(Span<float> input, Span<float> output, Span<float> weights, Span<float> bias, in RuntimeShape inShape, int outputChannels, ValueRange<float> fusedActivation)
        {
            var filterShape = new RuntimeShape(outputChannels, inShape[1], 3, 3);
            int outputIdx = 0;

            for (int batch = 0; batch < inShape[0]; batch++)
            {
                var inBatch = input.Slice(batch * inShape[1] * inShape[2] * inShape[3], inShape[1] * inShape[2] * inShape[3]);

                for (int oc = 0; oc < outputChannels; oc++)
                {
                    var wOC = weights.Slice(oc * filterShape[1] * 3 * 3, filterShape[1] * 3 * 3);

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

                                    for (int ic = 0; ic < inShape[1]; ic++)
                                    {
                                        float inV = inBatch[(ic * inShape[2] + inY) * inShape[3] + inX];
                                        float w = wOC[(ic * 3 + ky) * 3 + kx];

                                        value += inV * w;
                                    }
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
