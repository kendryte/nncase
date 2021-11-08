// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using Nncase.IR;
using F = Nncase.IR.F;
using TensorType = tflite.TensorType;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitConv2DTranspose(in tflite.Operator op)
        {
            var (input, weights) = GetInputExprs(op, 0, 1);
            var bias = GetInputExprs(op, 2);
            var options = op.BuiltinOptionsAsTransposeConvOptions();
            var inH = GetInputTensor(op, 0).Shape(2);
            var inW = GetInputTensor(op, 0).Shape(3);
            var fH = GetInputTensor(op, 1).Shape(2);
            var fW = GetInputTensor(op, 1).Shape(3);
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = 1;
            var dilationW = 1;
            var padH = GetWindowedPadding(inH, fH, strideH, dilationH, options.Padding == tflite.Padding.SAME);
            var padW = GetWindowedPadding(inW, fW, strideW, dilationW, options.Padding == tflite.Padding.SAME);
            var paddingValue = padH.Concat(padW).ToArray();
            var stride = Const.FromSpan<int>(new[] { strideH, strideW }, new[] { 2 });
            var dilation = Const.FromSpan<int>(new[] { dilationH, dilationW }, new[] { 2 });
            var padding = Const.FromSpan<int>(paddingValue, new[] { 2, 2 });
            var clamp = ValueRange<float>.Full;
            return F.Math.Clamp(
                F.NN.Conv2DTranspose(input, weights, bias, padding, stride, dilation, PadMode.Constant, 1),
                clamp.Min, clamp.Max);
        }
    }
}