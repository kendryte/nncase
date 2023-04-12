// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr ShapeIndexToInt64(Expr input, int index)
        {
            return F.Tensors.Cast(Util.ShapeIndex(input, index), DataTypes.Int64);
        }

        private Expr VisitSpaceToDepth(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var blockSize = GetIntAttribute(op, "blocksize");

            var shape0 = ShapeIndexToInt64(input, 0);
            var shape1 = ShapeIndexToInt64(input, 1);
            var shape2 = ShapeIndexToInt64(input, 2);
            var shape3 = ShapeIndexToInt64(input, 3);
            var beforeNewShape =
                F.Tensors.Stack(
                    new Tuple(shape0, shape1, shape2 / blockSize, blockSize, shape3 / blockSize, blockSize), 0);
            var afterNewShape =
            F.Tensors.Stack(
            new Tuple(shape0, shape1 * blockSize * blockSize, shape2 / blockSize, shape3 / blockSize), 0);
            var perm = new[] { 0, 3, 5, 1, 2, 4 };
            return F.Tensors.Reshape(
                F.Tensors.Transpose(
                    F.Tensors.Reshape(input, beforeNewShape),
                    perm),
                afterNewShape);
        }
    }
}
