using System;
using NetFabric.Hyperlinq;
using OrtKISharp;
using static Nncase.IR.F.Tensors;
using static OrtKISharp.TensorHelper;
namespace Nncase.Evaluator;

public static class EvaluatorUtil
{
    /// <summary>
    /// nncase pads format to onnx pads format
    /// </summary>
    /// <param name="pads"></param>
    /// <returns></returns>
    public static OrtKISharp.Tensor ToOnnxPadFormat(OrtKISharp.Tensor pads)
    {
        if (pads.Rank != 2)
        {
            throw new InvalidOperationException($"Pad's rank must be 2, but get {pads.Rank}");
        }
        return MakeOrtTensor(OrtKI.Transpose(pads, new long[] {1, 0}).ToArray<long>());
    }
}