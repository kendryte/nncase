// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitReduceWindow2D(in tflite.Operator op, ReduceOp reduceOp, float initValue)
        {
            var input = GetInputExprs(op, 0);
            input = F.Tensors.NHWCToNCHW(input);
            var option = op.BuiltinOptionsAsPool2DOptions();
            var (inH, inW) = Util.GetHW(input);
            var filterH = option.FilterHeight;
            var filterW = option.FilterWidth;
            var strideH = option.StrideH;
            var strideW = option.StrideW;
            var padH = TypeInference.GetWindowedPadding(inH, filterH, strideH, 1, option.Padding == tflite.Padding.SAME);
            var padW = TypeInference.GetWindowedPadding(inW, filterW, strideW, 1, option.Padding == tflite.Padding.SAME);
            var filter = Tensor.From<int>(new[] { filterH, filterW }, [2]);
            var stride = Tensor.From<int>(new[] { strideH, strideW }, [2]);
            var padding = TypeInference.ConcatPadding(padH, padW);
            return F.Tensors.NCHWToNHWC(
                F.NN.ReduceWindow2D(
                    reduceOp, input, initValue, filter, stride, padding, Tensor.From<long>(new long[] { 1, 1 }), false, false));
        }
    }
}
