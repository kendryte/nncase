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
            // when op set is 2, Pads and PadMode are a attr
            var (input, pads) = GetInputExprs(op, 0, 1);
            var padMode = GetPadMode(op); 
            var padValue = op.Input.Count == 3
                ? GetInputExpr(op, 2)
                : 0;
            return F.Tensors.Pad(input, pads, PadMode.Constant, padValue);
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