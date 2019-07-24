using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Targets.K210.IR;

namespace NnCase.Targets.K210.Kernels
{
    public static partial class K210Kernels
    {
        public static void Pool2D(Span<float> input, Span<float> output, in RuntimeShape inShape, KPUPoolType poolType)
        {
            var filter = KPUShapeUtility.GetKPUFilterSize(poolType);
            var stride = KPUShapeUtility.GetKPUFilterStride(poolType);
            var outH = KPUShapeUtility.GetKPUOutputSize(inShape[2], poolType);
            var outW = KPUShapeUtility.GetKPUOutputSize(inShape[3], poolType);

            int outIdx = 0;
            for (int oc = 0; oc < inShape[1]; oc++)
            {
                var inC = input.Slice(oc * inShape[2] * inShape[3]);

                for (int oy = 0; oy < outH; oy++)
                {
                    for (int ox = 0; ox < outW; ox++)
                    {
                        int inYOrigin = oy * stride;
                        int inXOrigin = ox * stride;
                        float value = 0;

                        switch (poolType)
                        {
                            case KPUPoolType.Pool_Bypass:
                                {
                                    int inY = inYOrigin;
                                    int inX = inXOrigin;

                                    value = inC[inY * inShape[3] + inX];
                                    break;
                                }
                            case KPUPoolType.Pool_Max_2_S2:
                            case KPUPoolType.Pool_Max_2_S1:
                            case KPUPoolType.Pool_Max_4_S4:
                                {
                                    value = float.MinValue;
                                    for (int ky = 0; ky < filter; ky++)
                                    {
                                        for (int kx = 0; kx < filter; kx++)
                                        {
                                            int inY = inYOrigin + ky;
                                            int inX = inXOrigin + kx;
                                            float inV;

                                            if (inY < 0 || inY >= inShape[2] || inX < 0 || inX >= inShape[3])
                                                inV = float.MinValue;
                                            else
                                                inV = inC[inY * inShape[3] + inX];

                                            value = Math.Max(value, inV);
                                        }
                                    }

                                    break;
                                }
                            case KPUPoolType.Pool_Mean_2_S2:
                            case KPUPoolType.Pool_Mean_2_S1:
                            case KPUPoolType.Pool_Mean_4_S4:
                                {
                                    value = 0;
                                    for (int ky = 0; ky < filter; ky++)
                                    {
                                        for (int kx = 0; kx < filter; kx++)
                                        {
                                            int inY = Math.Clamp(inYOrigin + ky, 0, inShape[2] - 1);
                                            int inX = Math.Clamp(inXOrigin + kx, 0, inShape[3] - 1);
                                            var inV = inC[inY * inShape[3] + inX];

                                            value += inV;
                                        }
                                    }

                                    value /= filter * filter;
                                    break;
                                }
                            case KPUPoolType.Pool_LeftTop_2_S2:
                            case KPUPoolType.Pool_LeftTop_4_S4:
                            case KPUPoolType.Pool_RightTop_2_S2:
                                {
                                    var kOff = KPUShapeUtility.GetKPUSelectPoolOffset(poolType);
                                    int inY = inYOrigin + kOff.h;
                                    int inX = inXOrigin + kOff.w;
                                    float inV;

                                    if (inY < 0 || inY >= inShape[2] || inX < 0 || inX >= inShape[3])
                                        inV = 0;
                                    else
                                        inV = inC[inY * inShape[3] + inX];

                                    value = inV;
                                    break;
                                }
                        }

                        output[outIdx++] = value;
                    }
                }
            }
        }
    }
}
