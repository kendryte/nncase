﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
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
            var newOutShape = new Shape(outShape[0], outShape[3], outShape[1], outShape[2]);
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
            var stride = Tensor.From<int>(new[] { strideH, strideW }, [2]);
            var dilation = Tensor.From<int>(new[] { dilationH, dilationW }, [2]);
            var oldWShape = weights.CheckedShape;
            var wShape = new Shape(oldWShape[0], oldWShape[3], oldWShape[1], oldWShape[2]);
            var padding = TypeInference.GetPaddings(newOutShape, wShape, stride, dilation, options.Padding == tflite.Padding.SAME, false);
            var clamp = ValueRange<float>.Full;

            var conv2DTranspose = F.NN.Conv2DTranspose(
                    F.Tensors.NHWCToNCHW(input),
                    F.Tensors.NHWCToNCHW(weights),
                    bias,
                    newOutShape.ToValueArrayExpr(),
                    stride,
                    padding,
                    Tensor.From<long>(new long[] { 0, 0, 0, 0 }),
                    dilation,
                    PadMode.Constant,
                    1);
            List<string> outputNames = new() { GetOutputTensor(op, 0).Name };
            conv2DTranspose.Metadata.OutputNames = outputNames;
            return F.Tensors.NCHWToNHWC(F.Math.Clamp(
                conv2DTranspose,
                clamp.Min,
                clamp.Max));
        }
    }
}
