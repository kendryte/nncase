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
            var bias = GetBias(op, input);
            var autoPad = GetStringAttribute(op, "auto_pad", "NOTSET");

            var dilation = GetDilationsAttribute(op);
            var group = GetIntAttribute(op, "group", 1);
            // if not present, should be inferred from input W
            var strides = GetStrideAttribute(op);
            var pads = AutoPad(op, autoPad, input, weights, strides.ToArray<long>(), dilation.ToArray<long>());
            return F.NN.Conv2D(input, weights, bias, strides, pads, dilation, PadMode.Constant, group);
        }
        
        // only used for 2D
        private Const GetPadsAttribute(NodeProto op)
        {
            return Const.FromSpan<long>(GetIntsAttribute(op, "pads", 0, 4), new Shape(2, 2));
        }

        private Const GetStrideAttribute(NodeProto op)
        {
            return Const.FromSpan<long>(GetIntsAttribute(op, "stride", 1, 4));
        }

        private Const GetDilationsAttribute(NodeProto op)
        {
            return Const.FromSpan<long>(GetIntsAttribute(op, "dilations", new[] {1, 1}));
        }
        
        private Expr GetBias(NodeProto op, Expr input)
        {
            return op.Input.Count > 2
                ? GetInputExpr(op, 2)
                : F.Tensors.Broadcast(0f, Util.ShapeIndex(input, 3));
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
