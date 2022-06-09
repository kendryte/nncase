// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using LanguageExt.UnsafeValueAccess;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitConv2DTranspose(NodeProto op)
        {
            var (input, weights) = GetInputExprs(op, 0, 1);
            var bias = GetBias(op, weights, true);
            var strides = GetStrideAttribute(op);
            var dilation = GetDilationsAttribute(op);
            var group = GetIntAttribute(op, "group", 1);
            var autoPad = GetStringAttribute(op, "auto_pad", "NOTSET");
            var outputPadding = GetIntsAttribute(op, "output_paddings", new[] { 0, 0, 0, 0 });
            var pads = AutoPad(op, autoPad, input, weights, strides.ToArray<long>(), dilation.ToArray<long>());

            var outShape = GetOptionIntsAttribute(op, "output_shape")
                .Match(
                    o => Tensor.FromSpan<long>(o),
                    () => GetOutputShape(input, weights,
                        strides.ToArray<long>(),
                        outputPadding,
                        pads,
                        dilation,
                        autoPad));

            return F.NN.Conv2DTranspose(input, weights, bias, outShape, strides,
                pads, Tensor.FromSpan<long>(outputPadding),
                Tensor.FromSpan<long>(dilation), PadMode.Constant, group);
        }

        Expr ComputeOutSize(Expr inputSize, Expr weightSize, long[] strides, long[] outPaddings, Expr paddings, long[] dilations, int offset)
        {
            return strides[offset] * (inputSize - 1)
                + outPaddings[offset]
                + (weightSize - 1)
                * dilations[offset] + 1 - Util.GetItem(paddings, offset) - Util.GetItem(paddings, offset + 2);
        }

        Expr GetOutputShape(Expr input, Expr weights, long[] strides, long[] outPadding, Expr paddings, long[] dilations, string autoPad)
        {
            var iN = Util.ShapeIndex(input, 0);
            var iC = Util.ShapeIndex(input, 1);
            var (iH, iW) = Util.GetHW(input);
            var oc = Util.ShapeIndex(weights, 0);
            var ic = Util.ShapeIndex(weights, 1);
            var (wH, wW) = Util.GetHW(input);
            var outShape = new[] { iN, iC };
            if (autoPad is "SAME_UPPER" or "SAME_LOWER")
            {
                outShape.Append(iH * oc);
                outShape.Append(iW * ic);
            }
            else
            {
                outShape.Append(ComputeOutSize(iH, wH, strides, outPadding, paddings, dilations, 0));
                outShape.Append(ComputeOutSize(iW, wW, strides, outPadding, paddings, dilations, 1));
            }
            return F.Tensors.Concat(new IR.Tuple(outShape), 0);
        }
    }
}
