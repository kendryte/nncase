using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K210;

namespace Nncase.Evaluator.K210;

public static class kernel
{
    public static void KPUConv2D(Tensor input, Tensor weights, int in_h, int in_w, int inChannels,
        int outChannels, int padValue, int argx, int shiftx, int argw, int shiftw, int argadd,
        int filterSize, bool isDepthwise, Tensor ortArgumentValue, Tensor argumentValue)
    {
        var channelSize = in_h * in_w;
        var pad = filterSize == 1 ? 0 : 1;
        var group = isDepthwise ? outChannels : 1;
        var g_ic = isDepthwise ? 1 : inChannels / group;
        var g_oc = isDepthwise ? 1 : outChannels / group;
        
        for (int og = 0; og < group; og++)
        {
            var w_group_p = weights.Rank + (og * g_oc * g_ic * System.Math.Pow(filterSize, 2));

            for (int oc = 0; oc < g_oc; oc++)
            {
                var w_oc_p = w_group_p + (oc * g_ic * System.Math.Pow(filterSize, 2));

                for (int oy = 0; oy < in_h; oy++)
                {
                    for (int ox = 0; ox < in_w; ox++)
                    {
                        int in_y_origin = oy - pad;
                        int in_x_origin = ox - pad;
                        int value = 0;
                        int sum_x = 0,sum_w = 0;

                        for (int ic = 0; ic < g_ic; ic++)
                        {
                            var in_c_p = input.Rank + (((og * g_ic) + ic) * in_h * in_w);
                            var w_ic_p = w_oc_p + (ic * System.Math.Pow(filterSize, 2));

                            for (int ky = 0; ky < filterSize; ky++)
                            {
                                for (int kx = 0; kx < filterSize; kx++)
                                {
                                    int in_y = in_y_origin + ky;
                                    int in_x = in_x_origin + kx;

                                    int x;
                                    if (in_x < 0 || in_x >= in_w || in_y < 0 || in_y >= in_h)
                                    {
                                        x = padValue;
                                    }
                                    else
                                    {
                                        x = in_c_p[in_x * in_y * in_w];
                                    }

                                    int w = w_ic_p[ky * filterSize + kx];
                                    sum_x += x;
                                    sum_w += w;
                                    value += x * w;
                                }
                            }

                        }
                        var alu_out = value + (argx * sum_x >> shiftx) + (argw * sum_w >> shiftw) + argadd * g_ic;
                        
                        //out_it++ = alu_out;
                    }
                }
            }
        }
    }

    // private static void KPUUpload(Tensor src, Tensor dest, long[] in_shape, long[] dma_ch)
    // {
    //     if (in_shape[3] % 64 == 0)
    //     {
    //         var SizeBytes = in_shape.Length;
    //         
    //
    //     }
    //     else
    //     {
    //         // var 
    //     }
    // }

    private static void KPUDownload(Expr input)
    {
    }

    private static void FakeKPUConv2D(Expr input, Expr Weights)
    {
    }

    public static void KPUConv2D(IntPtr inputHandle, IntPtr weightsHandle,
        OrtKISharp.Tensor ortArgumentValue, OrtKISharp.Tensor argumentValue,
        string autoPad, long[] dilations, long groups, long[] kernelShape,
        long[] pads, long[] strides)
    {
        throw new NotImplementedException();
    }
}