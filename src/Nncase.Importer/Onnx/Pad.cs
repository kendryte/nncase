// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitPad(in NodeProto op)
        {
            return GetOpSet(op) < 11
                ? PadV2(op)
                : PadV11(op);
        }

        private Expr PadV2(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var padMode = GetPadMode(op);
            var pads = GetIntsAttribute(op, "pads");
            var paddings = Const.FromSpan<long>(pads, new Shape(pads.Length / 2, 2));
            var value = GetFloatAttribute(op, "value", 0f);
            return F.Tensors.Pad(input, paddings, padMode, value);
        }
        
        private Expr PadV11(in NodeProto op)
        {
            // todo:pads shape
            var (input, pads) = GetInputExprs(op, 0, 1);
            var padMode = GetPadMode(op);
            var padValue = GetOptionInputExpr(op, 2, 0);
            return F.Tensors.Pad(input, pads, padMode, padValue);
        }
        
        private PadMode GetPadMode(in NodeProto op)
        {
            var mode = GetStringAttribute(op, "mode", "constant");
            return mode switch
            {
                "constant" => PadMode.Constant,
                "reflect" => PadMode.Reflect,
                "edge" => PadMode.Edge,
                _ => throw new NotSupportedException($"Not Supported pad mode:{mode}"),
            };
        }
    }
}