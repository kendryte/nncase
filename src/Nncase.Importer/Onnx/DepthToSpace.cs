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
            var blockSize = GetIntAttribute(op, "blocksize");
            var mode = GetStringAttribute(op, "mode", "DCR");

            var shape0 = ShapeIndexToInt64(input, 0);
            var shape1 = ShapeIndexToInt64(input, 1);
            var shape2 = ShapeIndexToInt64(input, 2);
            var shape3 = ShapeIndexToInt64(input, 3);
            var depth = shape1 / (blockSize * blockSize);
            var beforeNewShape = mode == "DCR"
                ? F.Tensors.Stack(new Tuple(shape0, blockSize, blockSize, depth, shape2, shape3), 0)
                : F.Tensors.Stack(new Tuple(shape0, depth, blockSize, blockSize, shape2, shape3), 0);
            var afterNewShape = F.Tensors.Stack(new Tuple(shape0, depth, shape2 * blockSize, shape3 * blockSize), 0);
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
