// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#if false
using System;
using Nncase.CostModel;
using Nncase.Evaluator.Math;
using Nncase.IR;
using Nncase.IR.K210;
using Nncase.IR.Math;

namespace Nncase.Evaluator.K210;

public static class kernel
{
    // public static IValue KPUConv2D(OrtKISharp.Tensor input, OrtKISharp.Tensor weights, int inH, int inW, int inChannels,
    // int outChannels, long[] padValue, IValue argX, IValue shiftX, IValue argW, IValue shiftW, IValue argAdd,
    // OrtKISharp.Tensor argumentValue, OrtKISharp.Tensor ortArgumentValue)
    // {
    //     throw new NotImplementedException();
    // }
    public static IValue KPUConv2D(Tensorflow.Tensor input, Tensorflow.Tensor weights, int in_h, int in_w, int inChannels,
        int outChannels, int padValue, int argx, int shiftx, int argw, int shiftw, int argadd,
        int filterSize, bool isDepthwise, KPUActivationParameters ortArgumentValue, KPUBatchNormParameters argumentValue)
    {
        // var out_it = workspace;
        var channelSize = in_h * in_w;
        var pad = filterSize == 1 ? 0 : 1;
        var group = isDepthwise ? outChannels : 1;
        var g_ic = isDepthwise ? 1 : inChannels / group;
        var g_oc = isDepthwise ? 1 : outChannels / group;

        for (int og = 0; og < group; og++)
        {
            // var w_group_p = weights.Rank + (og * g_oc * g_ic * System.Math.Pow(filterSize, 2));

            for (int oc = 0; oc < g_oc; oc++)
            {
                // var w_oc_p = w_group_p + (oc * g_ic * System.Math.Pow(filterSize, 2));

                for (int oy = 0; oy < in_h; oy++)
                {
                    for (int ox = 0; ox < in_w; ox++)
                    {
                        Int32 in_y_origin = oy - pad;
                        Int32 in_x_origin = ox - pad;
                        Int64 value = 0;
                        Int64 sum_x = 0, sum_w = 0;

                        for (int ic = 0; ic < g_ic; ic++)
                        {
                            // var in_c_p = input.Rank + (((og * g_ic) + ic) * in_h * in_w);
                            // var w_ic_p = w_oc_p + (ic * System.Math.Pow(filterSize, 2));

                            for (int ky = 0; ky < filterSize; ky++)
                            {
                                for (int kx = 0; kx < filterSize; kx++)
                                {
                                    Int32 in_y = in_y_origin + ky;
                                    Int32 in_x = in_x_origin + kx;

                                    uint x;
                                    if (in_x < 0 || in_x >= in_w || in_y < 0 || in_y >= in_h)
                                    {
                                        x = (uint)padValue;
                                    }
                                    else
                                    {
                                        x = (uint)input[in_y * in_w + in_x];
                                    }

                                    uint w = (uint)weights[ky * filterSize + kx];
                                    sum_x += x;
                                    sum_w += (int)w;
                                    value += (int)(x * w);
                                }
                            }
                        }

                        var alu_out = value + (argx * sum_x >> shiftx) + (argw * sum_w >> shiftw) + argadd * g_ic;
                    }
                }
            }
        }

        // bn act
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                var bn = argumentValue.Segments[oc];
                for (int i = 0; i < channelSize; i++)
                {
                    var value = bn.Mul >> bn.Shift + bn.Add;
                    var seg = ortArgumentValue.Segments[value];
                    var actValue = KPUUtility.carryShift((value - seg.StartX) * seg.Mul, seg.Shift) + seg.Add;
                    // IR.F.Math.Clamp(actValue, 0, 255);
                    IR.F.Math.Clamp(actValue, 0, 255);
                }
            }
        }

        return null;
    }

    private static void KPUDownload(Expr input)
    {
    }

    private static void FakeKPUConv2D(Expr input, Expr Weights)
    {
    }

}
#endif
