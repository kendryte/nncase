using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.K210;

namespace Nncase.Evaluator.K210;

public static class kernel
{
    public static void KPUConv2D(Tensor input, Tensor weights, Tensor ortArgumentValue,
        Tensor argumentValue, string auto_pad,
        long[] dilations, long group, long[] kernel_shape, long[] pads, long[] strides)
    {
        
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