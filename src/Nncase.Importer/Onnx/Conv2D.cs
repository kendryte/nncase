// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitConv2D(in NodeProto op)
        {
            var (input, weights) = GetInputExprs(op, 0, 1);
            var bias = GetBias(op, weights);
            var autoPad = GetStringAttribute(op, "auto_pad", "NOTSET");
            var dilation = GetDilationsAttribute(op).ToList();
            var group = GetIntAttribute(op, "group", 1);

            // if not present, should be inferred from input W
            var strides = GetStrideAttribute(op).ToArray<long>().ToList();

            var isConv1D = IsConv1D(weights);
            if (isConv1D)
            {
                dilation.Add(1);
                strides.Add(1);
                input = To4D(input);
                weights = To4D(weights);
            }

            var pads = AutoPad(op, autoPad, input, weights, strides.ToArray<long>(), dilation.ToArray(), isConv1D);
            pads.InferenceType();
            var conv = F.NN.Conv2D(input, weights, bias, strides.ToArray(), pads, dilation.ToArray(), PadMode.Constant, group);
            if (isConv1D)
            {
                conv = Squeeze(conv, new[] { 3 });
            }

            return conv;
        }

        private Call To4D(Expr input) => Unsqueeze(input, new[] { 3 });

        private bool IsConv1D(Expr weights)
        {
            bool conv1d = false;
            weights.InferenceType();
            var weightsRank = weights.CheckedShape.Rank;
            switch (weightsRank)
            {
                case 3:
                    conv1d = true;
                    break;
                case 4:
                    break;
                default:
                    throw new NotSupportedException($"only support 1d and 2d, but get weights rank {weightsRank}");
            }

            return conv1d;
        }

        private Expr GetPadsAttribute(NodeProto op, bool isConv1D = false)
        {
            var paddings = GetIntsAttribute(op, "pads", 0, 4);
            if (isConv1D)
            {
                paddings = new[] { paddings[0], 0, paddings[1], 0 };
            }

            return ToNncasePadFormat(paddings);
        }

        private Tensor GetStrideAttribute(NodeProto op)
        {
            return Tensor.From<long>(GetIntsAttribute(op, "strides", 1, 2));
        }

        private long[] GetDilationsAttribute(NodeProto op)
        {
            return GetIntsAttribute(op, "dilations", new[] { 1, 1 });
        }

        private Expr GetBias(NodeProto op, Expr weights, bool isConvTranspose = false, long groups = 1)
        {
            var biasSizeIndex = isConvTranspose ? 1 : 0;
            return op.Input.Count > 2
                ? GetInputExpr(op, 2)
                : F.Tensors.Expand(0f, Util.ShapeIndex(weights, biasSizeIndex) * groups);
        }

        private Expr AutoPad(NodeProto op, string autoPad, Expr input, Expr weights, long[] strides, long[] dilation, bool isConv1D = false) => autoPad switch
        {
            "NOTSET" => GetPadsAttribute(op, isConv1D),
            "SAME_UPPER" => Util.GetPaddings(input, weights, strides, dilation, true),
            "SAME_LOWER" => Util.GetPaddings(input, weights, strides, dilation, true, true),
            "VALID" => GetPadsAttribute(op, isConv1D),

            // when VALID, I'm not sure this is correct
            // in onnx doc, not spec when VALID value
            // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
            _ => throw new InvalidDataException($"invalid AutoPad Value: {autoPad}"),
        };
    }
}
