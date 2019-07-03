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


            for (int batch = 0; batch < inShape[0]; batch++)
            {
                for (int og = 0; og < groups; og++)
                {
                    for (int oc = 0; oc < goc; oc++)
                    {
                        for (int oy = 0; oy < outH; oy++)
                        {
                            for (int ox = 0; ox < outW; ox++)
                            {
                                int inYOrigin = (oy * strideH) - paddingH.Before;
                                int inXOrigin = (ox * strideW) - paddingW.Before;
                                int filterYStart = Math.Max(0, -inYOrigin);
                                int filterYEnd = Math.Min(filterH, inShape[2] - inYOrigin);
                                int filterXSstart = Math.Max(0, -inXOrigin);
                                int filterXEnd = Math.Min(filterW, inShape[3] - inXOrigin);
                                float value = bias[oc];

                                for (int ky = filterYStart; ky < filterYEnd; ky++)
                                {
                                    for (int kx = filterXSstart; kx < filterXEnd; kx++)
                                    {
                                        int inY = inYOrigin + dilationH * ky;
                                        int inX = inXOrigin + dilationW * kx;

                                        for (int ic = 0; ic < gic; ic++)
                                        {
                                            float in_v = input[Offset(inShape, new RuntimeShape(batch, og * gic + ic, inY, inX))];
                                            float w = weights[Offset(filterShape, new RuntimeShape(og * goc + oc, ic, ky, kx))];

                                            value += in_v * w;
                                        }
                                    }
                                }

                                output[Offset(outShape, new RuntimeShape(batch, og * goc + oc, oy, ox))] = ApplyActivation(value, fusedActivation);
                            }
                        }
                    }
                }
            }
        }
    }
}
