// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using LanguageExt.UnsafeValueAccess;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.Tensors;
using static Nncase.ResizeModeHelper;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitUpsample(in NodeProto op)
        {
            var (input, scales) = GetInputExprs(op, 0, 1);
            var inputShape = F.Tensors.ShapeOf(input);
            var roi = Enumerable.Repeat((Expr)0f, input.CheckedShape.Rank).ToList();
            for (var i = 0; i < input.CheckedShape.Rank; i++)
            {
                roi.Add(IR.F.Tensors.Cast(inputShape[i], DataTypes.Float32));
            }

            var mode = ImageResizeMode.NearestNeighbor;
            return F.Imaging.ResizeImage(mode, input, Stack(new IR.Tuple(roi.ToArray()), 0), ComputeNewSizes(input, scales));
        }
    }
}
