// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Onnx;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;

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
            var paddings = GetIntsAttribute(op, "pads");
            var pads = Tensor.From<long>(paddings, new[] { 2, 4 });
            var value = GetFloatAttribute(op, "value", 0f);
            return Pad(input, ToNncasePadFormat(pads), padMode, value);
        }

        // `pads` should be a 1D tensor of shape [2 * input_rank].
        // `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...]
        private Expr PadV11(in NodeProto op)
        {
            var (input, pads) = GetInputExprs(op, 0, 1);
            var padMode = GetPadMode(op);

            // GetInputExpr will get a Tensor with shape [1], but padValue is a scalar
            var padValue = GetOptionInputExpr(op, 2)
                .Match(
                    x => SliceIndex(Stack(new IR.Tuple(x), 0), 0),
                    () => 0f);
            return Pad(input, ToNncasePadFormat(pads), padMode, padValue);
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
