// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using tflite;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;
using TensorType = tflite.TensorType;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitPad(in tflite.Operator op)
        {
            var (input, paddings) = GetInputExprs(op, 0, 1);
            input = F.Tensors.NHWCToNCHW(input);
            paddings = F.Tensors.PaddingNHWCToNCHW(paddings, input.CheckedShape.Rank);
            var padValue = GetInputTensor(op, 0).Type switch
            {
                TensorType.FLOAT32 => 0.0f,
                TensorType.INT8 => (sbyte)0,
                TensorType.UINT8 => (byte)128,
                _ => throw new NotSupportedException("Unsupported Constant Pad Value"),
            };

            return F.Tensors.NCHWToNHWC(F.NN.Pad(input, paddings, PadMode.Constant, padValue), input.CheckedShape.Rank);
        }

        private Expr VisitPadV2(in tflite.Operator op)
        {
            var (input, paddings) = GetInputExprs(op, 0, 1);
            input = F.Tensors.NHWCToNCHW(input);
            paddings = F.Tensors.PaddingNHWCToNCHW(paddings, input.CheckedShape.Rank);
            var padValue = GetInputExprs(op, 2);
            return F.Tensors.NCHWToNHWC(F.NN.Pad(input, paddings, PadMode.Constant, padValue), input.CheckedShape.Rank);
        }

        private Expr VisitMirrorPad(in tflite.Operator op)
        {
            var (input, paddings) = GetInputExprs(op, 0, 1);
            input = F.Tensors.NHWCToNCHW(input);
            paddings = F.Tensors.PaddingNHWCToNCHW(paddings, input.CheckedShape.Rank);
            var padMode = op.BuiltinOptionsAsMirrorPadOptions().Mode switch
            {
                tflite.MirrorPadMode.REFLECT => PadMode.Reflect,
                tflite.MirrorPadMode.SYMMETRIC => PadMode.Symmetric,
                _ => throw new NotSupportedException("Unsupported Mirror Pad Mode"),
            };

            return F.Tensors.NCHWToNHWC(F.NN.Pad(input, paddings, padMode, 0.0f), input.CheckedShape.Rank);
        }
    }
}
