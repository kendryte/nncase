// Copyright (c) Canaan Inc. All rights reserved.
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
            var input = GetInputExpr(op, 0);
            var blockSize = (int)GetIntAttribute(op, "blocksize");
            var mode = GetStringAttribute(op, "mode", "DCR");

            var shape0 = Util.ShapeIndex(input, 0);
            var shape1 = Util.ShapeIndex(input, 1);
            var shape2 = Util.ShapeIndex(input, 2);
            var shape3 = Util.ShapeIndex(input, 3);
            var depth = shape1 / (blockSize * blockSize);
            var beforeNewShape = mode == "DCR"
                ? F.Tensors.Concat(new Tuple(shape0, blockSize, blockSize, depth, shape2, shape3), 0)
                : F.Tensors.Concat(new Tuple(shape0, depth, blockSize, blockSize, shape2, shape3), 0);
            var afterNewShape = F.Tensors.Concat(new Tuple(shape0, depth, shape2 * blockSize, shape3 * blockSize), 0);
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