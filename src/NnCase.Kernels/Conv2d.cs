using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void Conv2D(Span<float> input, Span<float> output, Span<float> weights, Span<float> bias, in RuntimeShape inShape, int groups, int outputChannels, int filterH, int filterW, int strideH, int strideW, int dilationH, int dilationW, Padding paddingH, Padding paddingW, ValueRange<float> fusedActivation)
        {
            var outH = GetWindowedOutputSize(inShape[2], filterH, strideH, dilationH, paddingH);
            var outW = GetWindowedOutputSize(inShape[3], filterW, strideW, dilationW, paddingW);
            var gic = inShape[1] / groups;
            var goc = outputChannels / groups;
            var filterShape = new RuntimeShape(outputChannels, gic, filterH, filterW);
            var outShape = new RuntimeShape(inShape[0], outputChannels, outH, outW);
            int outputIdx = 0;

            for (int batch = 0; batch < inShape[0]; batch++)
            {
                var inBatch = input.Slice(batch * inShape[1] * inShape[2] * inShape[3], inShape[1] * inShape[2] * inShape[3]);

                for (int og = 0; og < groups; og++)
                {
                    var inGroup = inBatch.Slice(og * gic * inShape[2] * inShape[3], gic * inShape[2] * inShape[3]);
                    var wGroup = weights.Slice(og * goc * gic * filterShape[2] * filterShape[3], goc * gic * filterShape[2] * filterShape[3]);

                    for (int oc = 0; oc < goc; oc++)
                    {
                        var wOC = wGroup.Slice(oc * gic * filterShape[2] * filterShape[3], gic * filterShape[2] * filterShape[3]);

                        for (int oy = 0; oy < outH; oy++)
                        {
                            for (int ox = 0; ox < outW; ox++)
                            {
                                int inYOrigin = (oy * strideH) - paddingH.Before;
                                int inXOrigin = (ox * strideW) - paddingW.Before;
                                int filterYStart = Math.Max(0, (-inYOrigin + dilationH - 1) / dilationH);
                                int filterYEnd = Math.Min(filterH, (inShape[2] - inYOrigin + dilationH - 1) / dilationH);
                                int filterXSstart = Math.Max(0, (-inXOrigin + dilationW - 1) / dilationW);
                                int filterXEnd = Math.Min(filterW, (inShape[3] - inXOrigin + dilationW - 1) / dilationW);
                                float value = bias[oc];

                                for (int ky = filterYStart; ky < filterYEnd; ky++)
                                {
                                    for (int kx = filterXSstart; kx < filterXEnd; kx++)
                                    {
                                        int inY = inYOrigin + dilationH * ky;
                                        int inX = inXOrigin + dilationW * kx;

                                        for (int ic = 0; ic < gic; ic++)
                                        {
                                            float inV = inGroup[(ic * inShape[2] + inY) * inShape[3] + inX];
                                            float w = wOC[(ic * filterShape[2] + ky) * filterShape[3] + kx];

                                            value += inV * w;
                                        }
                                    }
                                }

                                output[outputIdx++] = ApplyActivation(value, fusedActivation);
                            }
                        }
                    }
                }
            }
        }
    }
}
