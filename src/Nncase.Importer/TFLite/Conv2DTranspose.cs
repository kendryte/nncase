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
            var strideH = options.StrideH;
            var strideW = options.StrideW;
            var dilationH = 1;
            var dilationW = 1;
            var stride = Tensor.From<int>(new[] { strideH, strideW }, new[] { 2 });
            var dilation = Tensor.From<int>(new[] { dilationH, dilationW }, new[] { 2 });
            var oldWShape = F.Tensors.ShapeOf(weights);
            var wShape = F.Tensors.Stack(new IR.Tuple(oldWShape[0], oldWShape[3], oldWShape[1], oldWShape[2]), 0);
            var padding = F.ShapeExpr.GetPaddings(F.Tensors.Stack(new IR.Tuple(newOutShape), 0), wShape, stride,
            dilation, options.Padding == tflite.Padding.SAME, false);
            var clamp = ValueRange<float>.Full;

            return F.Tensors.NCHWToNHWC(F.Math.Clamp(
                F.NN.Conv2DTranspose(
                    F.Tensors.NHWCToNCHW(input),
                    F.Tensors.NHWCToNCHW(weights),
                    bias,
                    IR.F.Tensors.Stack(new IR.Tuple(newOutShape), 0),
                    stride,
                    padding,
                    Tensor.From<long>(new long[] { 0, 0, 0, 0 }),
                    dilation,
                    PadMode.Constant,
                    1),
                clamp.Min,
                clamp.Max));
        }
    }
}
