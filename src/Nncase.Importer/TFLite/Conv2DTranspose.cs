// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
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
            var newOutShape = new[] { outShape[0], outShape[3], outShape[1], outShape[2] };
            var (input, weights) = GetInputExprs(op, 2, 1);
            Expr bias;
            if (op.InputsLength > 3)
            {
                bias = GetInputExprs(op, 3);
            }
            else
            {
                var oc = IR.F.Tensors.ShapeOf(weights)[0];
                bias = IR.F.Tensors.Expand(new[] { 0f }, IR.F.Tensors.StackScalar(oc));
            }

            var options = op.BuiltinOptionsAsTransposeConvOptions();
            var (_, _) = Util.GetHW(input);
            var (fH, fW) = Util.GetHW(weights, true);
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = 1;
            var dilationW = 1;
            var padH = Util.GetWindowedPadding(newOutShape[2], fH, strideH, dilationH, options.Padding == tflite.Padding.SAME);
            var padW = Util.GetWindowedPadding(newOutShape[3], fW, strideW, dilationW, options.Padding == tflite.Padding.SAME);
            var stride = Tensor.From<int>(new[] { strideH, strideW }, new[] { 2 });
            var dilation = Tensor.From<int>(new[] { dilationH, dilationW }, new[] { 2 });
            var padding = Util.ConcatPadding(padH, padW);
            var clamp = ValueRange<float>.Full;

            var conv2DTranspose = F.NN.Conv2DTranspose(
                    F.Tensors.NHWCToNCHW(input),
                    F.Tensors.NHWCToNCHW(weights),
                    bias,
                    IR.F.Tensors.Stack(new IR.Tuple(newOutShape), 0),
                    stride,
                    padding,
                    Tensor.From<long>(new long[] { 0, 0, 0, 0 }),
                    dilation,
                    PadMode.Constant,
                    1);
            List<string> outputNames = new() { GetInputTensor(op, 0).Name };
            conv2DTranspose.Metadata.OutputNames = outputNames;
            return F.Tensors.NCHWToNHWC(F.Math.Clamp(
                conv2DTranspose,
                clamp.Min,
                clamp.Max));
        }
    }
}
