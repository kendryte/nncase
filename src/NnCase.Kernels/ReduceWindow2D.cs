using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Kernels
{
    public static partial class DefaultKernels
    {
        public static void ReduceWindow2D(Span<float> input, Span<float> output, float initialValue, in RuntimeShape inShape, int filterH, int filterW, int strideH, int strideW, int dilationH, int dilationW, Padding paddingH, Padding paddingW, ValueRange<float> fusedActivation, Func<float, float, float> binaryOp, Func<float, int, float> windowOp)
        {
            var outH = GetWindowedOutputSize(inShape[2], filterH, strideH, dilationH, paddingH);
            var outW = GetWindowedOutputSize(inShape[3], filterW, strideW, dilationW, paddingW);
            var outShape = new RuntimeShape(inShape[0], inShape[1], outH, outW);

            for (int batch = 0; batch < inShape[0]; batch++)
            {
                for (int oc = 0; oc < inShape[1]; oc++)
                {
                    for (int oy = 0; oy < outH; oy++)
                    {
                        for (int ox = 0; ox < outW; ox++)
                        {
                            int inYOrigin = (oy * strideH) - paddingH.Before;
                            int inXOrigin = (ox * strideW) - paddingW.Before;
                            int filterYStart = Math.Max(0, (-inYOrigin + dilationH - 1) / dilationH);
                            int filterYEnd = Math.Min(filterH, inShape[2] - inYOrigin);
                            int filterXSstart = Math.Max(0, (-inXOrigin + dilationW - 1) / dilationW);
                            int filterXEnd = Math.Min(filterW, inShape[3] - inXOrigin);
                            float value = initialValue;
                            int kernelCount = 0;

                            for (int ky = filterYStart; ky < filterYEnd; ky++)
                            {
                                for (int kx = filterXSstart; kx < filterXEnd; kx++)
                                {
                                    int inY = inYOrigin + dilationH * ky;
                                    int inX = inXOrigin + dilationW * kx;

                                    float inV = input[Offset(inShape, new RuntimeShape(batch, oc, inY, inX))];
                                    value = binaryOp(value, inV);
                                    kernelCount++;
                                }
                            }

                            value = windowOp(value, kernelCount);
                            output[Offset(outShape, new RuntimeShape(batch, oc, oy, ox))] = ApplyActivation(value, fusedActivation);
                        }
                    }
                }
            }
        }
    }
}
