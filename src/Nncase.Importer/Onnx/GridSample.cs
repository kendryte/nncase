// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using LanguageExt.UnsafeValueAccess;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.GridSampleModeHelper;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitGridSample(in NodeProto op)
        {
            var opSet = GetOpSet(op);
            if (opSet > 16)
            {
                throw new NotSupportedException("GridSample opset > 16 is not supported");
            }

            return GridSample(op);
        }

        private Expr GridSample(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var grid = GetInputExpr(op, 1);
            var alignCorners = ParseGridSampleAlignCorners(
                (int)GetIntAttribute(op, "align_corners", 0));
            var mode = ParseGridSampleMode(GetStringAttribute(op, "mode", "bilinear"));
            var paddingMode = ParseGridSamplePaddingMode(GetStringAttribute(op, "padding_mode", "zeros"));
            return F.NN.GridSample(input, grid, alignCorners, mode, paddingMode);
        }
    }
}
