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

        private Expr TransposePadding(Expr padding)
        {
            return Transpose(padding, new[] { 1, 0 });
        }

        private Expr PadV2(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var padMode = GetPadMode(op);
            var pads = GetIntsAttribute(op, "pads");
            var paddings = Const.FromSpan<long>(pads, new Shape(2, pads.Length / 2));
            var value = GetFloatAttribute(op, "value", 0f);
            return Pad(input, TransposePadding(paddings), padMode, value);
        }

        private Expr ReshapePadding(Expr pads)
        {
            return Reshape(pads,
                Concat(
                    new IR.Tuple(
                        new[] { 2 },
                        ShapeOf(pads) / 2),
                    0));
        }

        private Expr PadV11(in NodeProto op)
        {
            // todo:pads shape
            var (input, pads) = GetInputExprs(op, 0, 1);
            var reshapePads = ReshapePadding(pads);
            var padMode = GetPadMode(op);

            // GetInputExpr will get a Tensor with shape [1], but padValue is a scalar
            var padValue = GetOptionInputExpr(op, 2)
                .Match(
                    x => SliceIndex(x, 0),
                    () => 0);
            return Pad(input, TransposePadding(reshapePads), padMode, padValue);
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