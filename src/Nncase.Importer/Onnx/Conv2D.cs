// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using System.Linq;
using Nncase.IR;
using Onnx;
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

            var dilation = GetDilationsAttribute(op);
            var group = GetIntAttribute(op, "group", 1);

            // if not present, should be inferred from input W
            var strides = GetStrideAttribute(op);
            var pads = AutoPad(op, autoPad, input, weights, strides.ToArray<long>(), dilation.ToArray<long>());
            return F.NN.Conv2D(input, weights, bias, strides, pads, dilation, PadMode.Constant, group);
        }

        // only used for 2D
        private Expr GetPadsAttribute(NodeProto op)
        {
            // h before, w before, h after, w after
            // todo: padding size == 2?
            var paddings = GetIntsAttribute(op, "pads", 0, 4);
            var padsValue = new long[paddings.Length];
            var count = paddings.Length / 2;
            for (int i = 0; i < count; ++i)
            {
                padsValue[2 * i] = paddings[count - 1 - i];
                padsValue[2 * i + 1] = paddings[paddings.Length - 1 - i];
            }

            var paddingsValue = new long[] { paddings[1], paddings[3], paddings[0], paddings[2] };
            return Const.FromSpan<long>(padsValue, new Shape(count, 2));
        }

        private Const GetStrideAttribute(NodeProto op)
        {
            return Const.FromSpan<long>(GetIntsAttribute(op, "strides", 1, 2));
        }

        private Const GetDilationsAttribute(NodeProto op)
        {
            return Const.FromSpan<long>(GetIntsAttribute(op, "dilations", new[] { 1, 1 }));
        }

        private Expr GetBias(NodeProto op, Expr weights, bool isConvTranspose = false)
        {
            var biasSizeIndex = isConvTranspose ? 1 : 0;
            return op.Input.Count > 2
                ? GetInputExpr(op, 2)
                : F.Tensors.Expand(0f, Util.ShapeIndex(weights, biasSizeIndex));
        }

        private Expr AutoPad(NodeProto op, string autoPad, Expr input, Expr weights,
            long[] strides, long[] dilation) => autoPad switch
            {
                "NOTSET" => GetPadsAttribute(op),
                "SAME_UPPER" => Util.GetPaddings(input, weights, strides, dilation, true),
                "SAME_LOWER" => Util.GetPaddings(input, weights, strides, dilation, true, true),
                "VALID" => GetPadsAttribute(op),

                // when VALID, I'm not sure this is correct
                // in onnx doc, not spec when VALID value
                // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
                _ => throw new InvalidDataException($"invalid AutoPad Value: {autoPad}"),
            };
    }
}
