// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using Nncase.IR;
using F = Nncase.IR.F;
using TensorType = tflite.TensorType;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitConv2DTranspose(in tflite.Operator op)
        {
            var outShape = GetInputExprs(op, 0);
            var (input, weights) = GetInputExprs(op, 1, 2);
            var bias = GetInputExprs(op, 2);
            var options = op.BuiltinOptionsAsTransposeConvOptions();
            var (inH, inW) = Util.GetHW(input);
            var (fH, fW) = Util.GetHW(weights);
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = 1;
            var dilationW = 1;
            var padH = Util.GetWindowedPadding(inH, fH, strideH, dilationH, options.Padding == tflite.Padding.SAME);
            var padW = Util.GetWindowedPadding(inW, fW, strideW, dilationW, options.Padding == tflite.Padding.SAME);
            var stride = Tensor.From<int>(new[] { strideH, strideW }, new[] { 2 });
            var dilation = Tensor.From<int>(new[] { dilationH, dilationW }, new[] { 2 });
            var padding = Util.ConcatPadding(padH, padW);
            var clamp = ValueRange<float>.Full;
            return F.Tensors.NCHWToNHWC(F.Math.Clamp(
                F.NN.Conv2DTranspose(F.Tensors.NHWCToNCHW(input), F.Tensors.NHWCToNCHW(weights), bias,
                    F.Tensors.NHWCToNCHW(outShape), stride, padding, Tensor.From<long>(new long[] { 0, 0, 0, 0 }),
                    dilation, PadMode.Constant, 1), clamp.Min, clamp.Max));
        }
    }
}
