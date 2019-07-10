using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using NnCase.Kernels;

namespace NnCase.Targets.CPU.Kernels
{
    public static partial class CPUKernels
    {
        public static void QuantizedConv2D(Span<byte> input, Span<byte> output, Span<byte> weights, Span<int> bias, in RuntimeShape inShape, int outputChannels, int filterH, int filterW, int strideH, int strideW, int dilationH, int dilationW, Padding paddingH, Padding paddingW, int inputOffset, int filterOffset, int outputMul, int outputShift, int outputOffset)
        {
            var outH = DefaultKernels.GetWindowedOutputSize(inShape[1], filterH, strideH, dilationH, paddingH);
            var outW = DefaultKernels.GetWindowedOutputSize(inShape[2], filterW, strideW, dilationW, paddingW);
            var filterShape = new RuntimeShape(outputChannels, filterH, filterW, inShape[3]);
            int outputIdx = 0;

            for (int batch = 0; batch < inShape[0]; batch++)
            {
                var inBatch = input.Slice(batch * inShape[1] * inShape[2] * inShape[3], inShape[1] * inShape[2] * inShape[3]);

                for (int oy = 0; oy < outH; oy++)
                {
                    for (int ox = 0; ox < outW; ox++)
                    {
                        int inYOrigin = (oy * strideH) - paddingH.Before;
                        int inXOrigin = (ox * strideW) - paddingW.Before;
                        int filterYStart = Math.Max(0, (-inYOrigin + dilationH - 1) / dilationH);
                        int filterYEnd = Math.Min(filterH, (inShape[1] - inYOrigin + dilationH - 1) / dilationH);
                        int filterXSstart = Math.Max(0, (-inXOrigin + dilationW - 1) / dilationW);
                        int filterXEnd = Math.Min(filterW, (inShape[2] - inXOrigin + dilationW - 1) / dilationW);

                        for (int oc = 0; oc < outputChannels; oc++)
                        {
                            var wOC = weights.Slice(oc * filterShape[1] * filterShape[2] * filterShape[3], filterShape[1] * filterShape[2] * filterShape[3]);
                            int value = bias[oc];

                            for (int ky = filterYStart; ky < filterYEnd; ky++)
                            {
                                for (int kx = filterXSstart; kx < filterXEnd; kx++)
                                {
                                    int inY = inYOrigin + dilationH * ky;
                                    int inX = inXOrigin + dilationW * kx;

                                    var inPix = inBatch.Slice((inY * inShape[2] + inX) * inShape[3], filterShape[3]);
                                    var wPix = wOC.Slice((ky * filterShape[2] + kx) * filterShape[3], filterShape[3]);

                                    for (int ic = 0; ic < filterShape[3]; ic++)
                                    {
                                        int inV = inPix[ic] - inputOffset;
                                        int w = wPix[ic] - filterOffset;

                                        value += inV * w;
                                    }
                                }
                            }

                            value = DefaultKernels.MulAndCarryShift(value, outputMul, outputShift) + outputOffset;
                            output[outputIdx++] = (byte)Math.Clamp(value, 0, 255);
                        }
                    }
                }
            }
        }

        public static void QuantizedDepthwiseConv2D(Span<byte> input, Span<byte> output, Span<byte> weights, Span<int> bias, in RuntimeShape inShape, int filterH, int filterW, int strideH, int strideW, int dilationH, int dilationW, Padding paddingH, Padding paddingW, int inputOffset, int filterOffset, int outputMul, int outputShift, int outputOffset)
        {
            var outH = DefaultKernels.GetWindowedOutputSize(inShape[1], filterH, strideH, dilationH, paddingH);
            var outW = DefaultKernels.GetWindowedOutputSize(inShape[2], filterW, strideW, dilationW, paddingW);
            var filterShape = new RuntimeShape(inShape[3], filterH, filterW, 1);
            int outputIdx = 0;

            for (int batch = 0; batch < inShape[0]; batch++)
            {
                var inBatch = input.Slice(batch * inShape[1] * inShape[2] * inShape[3], inShape[1] * inShape[2] * inShape[3]);

                for (int oy = 0; oy < outH; oy++)
                {
                    for (int ox = 0; ox < outW; ox++)
                    {
                        int inYOrigin = (oy * strideH) - paddingH.Before;
                        int inXOrigin = (ox * strideW) - paddingW.Before;
                        int filterYStart = Math.Max(0, (-inYOrigin + dilationH - 1) / dilationH);
                        int filterYEnd = Math.Min(filterH, (inShape[1] - inYOrigin + dilationH - 1) / dilationH);
                        int filterXSstart = Math.Max(0, (-inXOrigin + dilationW - 1) / dilationW);
                        int filterXEnd = Math.Min(filterW, (inShape[2] - inXOrigin + dilationW - 1) / dilationW);

                        for (int oc = 0; oc < filterShape[0]; oc++)
                        {
                            var wOC = weights.Slice(oc * filterShape[1] * filterShape[2], filterShape[1] * filterShape[2]);
                            int value = bias[oc];

                            for (int ky = filterYStart; ky < filterYEnd; ky++)
                            {
                                for (int kx = filterXSstart; kx < filterXEnd; kx++)
                                {
                                    int inY = inYOrigin + dilationH * ky;
                                    int inX = inXOrigin + dilationW * kx;

                                    var inPix = inBatch.Slice((inY * inShape[2] + inX) * inShape[3], inShape[3]);
                                    var wPix = wOC.Slice((ky * filterShape[2] + kx) * filterShape[3], 1);

                                    int inV = inPix[oc] - inputOffset;
                                    int w = wPix[0] - filterOffset;

                                    value += inV * w;
                                }
                            }

                            value = DefaultKernels.MulAndCarryShift(value, outputMul, outputShift) + outputOffset;
                            output[outputIdx++] = (byte)Math.Clamp(value, 0, 255);
                        }
                    }
                }
            }
        }
    }
}
