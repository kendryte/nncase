﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitDepthToSpace(in NodeProto op)
        {
            var input = GetInputExpr<Expr>(op, 0);
            var blockSize = GetIntAttribute(op, "blocksize");
            var mode = GetStringAttribute(op, "mode", "DCR");

            var shape0 = Util.ShapeIndex(input, 0);
            var shape1 = Util.ShapeIndex(input, 1);
            var shape2 = Util.ShapeIndex(input, 2);
            var shape3 = Util.ShapeIndex(input, 3);
            var depth = shape1 / (blockSize * blockSize);
            var beforeNewShape = mode == "DCR"
                ? new RankedShape(shape0, blockSize, blockSize, depth, shape2, shape3)
                : new RankedShape(shape0, depth, blockSize, blockSize, shape2, shape3);
            var afterNewShape = new RankedShape(shape0, depth, shape2 * blockSize, shape3 * blockSize);
            var perm = mode == "DCR"
                ? new[] { 0, 3, 4, 1, 5, 2 }
                : new[] { 0, 1, 4, 2, 5, 3 };
            return F.Tensors.Reshape(
                F.Tensors.Transpose(
                    F.Tensors.Reshape(input, beforeNewShape),
                    perm),
                afterNewShape);
        }
    }
}
