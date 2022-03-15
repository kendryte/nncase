// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitReduceWindow2D(in tflite.Operator op, ReduceOp reduceOp, float initValue)
        {
            var input = GetInputExprs(op, 0);
            var option = op.BuiltinOptionsAsPool2DOptions();
            var (inH, inW) = Util.GetHW(input);
            var filterH = option.FilterHeight;
            var filterW = option.FilterWidth;
            var strideH = option.StrideH;
            var strideW = option.StrideW;
            var padH = Util.GetWindowedPadding(inH, filterH, strideH, 1, option.Padding == tflite.Padding.SAME);
            var padW = Util.GetWindowedPadding(inW, filterW, strideW, 1, option.Padding == tflite.Padding.SAME);
            var filter = Tensor.FromSpan<int>(new[] { filterH, filterW }, new[] { 2 });
            var stride = Tensor.FromSpan<int>(new[] { strideH, strideW }, new[] { 2 });
            var padding = Util.ConcatPadding(padH, padW);
            return F.Tensors.NCHWToNHWC(
                F.NN.ReduceWindow2D(
                    reduceOp, F.Tensors.NHWCToNCHW(input), initValue, filter, stride, padding, Tensor.FromSpan<long>(new long[] {1, 1}),false, false));
        }
    }
}