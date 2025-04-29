﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitSpaceToDepth(in NodeProto op)
        {
            var input = GetInputExpr<Expr>(op, 0);
            var blockSize = GetIntAttribute(op, "blocksize");

            var shape0 = Util.ShapeIndex(input, 0);
            var shape1 = Util.ShapeIndex(input, 1);
            var shape2 = Util.ShapeIndex(input, 2);
            var shape3 = Util.ShapeIndex(input, 3);
            var beforeNewShape = new RankedShape(shape0, shape1, shape2 / blockSize, blockSize, shape3 / blockSize, blockSize);
            var afterNewShape = new RankedShape(shape0, shape1 * blockSize * blockSize, shape2 / blockSize, shape3 / blockSize);
            var perm = new[] { 0, 3, 5, 1, 2, 4 };
            return F.Tensors.Reshape(
                F.Tensors.Transpose(
                    F.Tensors.Reshape(input, beforeNewShape),
                    perm),
                afterNewShape);
        }
    }
}
