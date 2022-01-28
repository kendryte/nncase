// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitFlatten(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var axis = GetIntAttribute(op, "axis", 1);
            return F.Tensors.Flatten(input, axis);

            // var inputShape = F.Tensors.ShapeOp(input);
            // var beforeFlatten = F.Tensors.Slice(inputShape, 0, axis, 1);
            // var afterFlatten = F.Tensors.Slice(inputShape, axis, F.Tensors.ShapeOp(inputShape), 1);
            // var flattenedAxis = F.Tensors.Reduce(ReduceOp.Sum, afterFlatten, 0, 0, false);
            // return F.Tensors.Reshape(input,
            //     F.Tensors.Concat(new Tuple(beforeFlatten, flattenedAxis), 0));
        }
    }
}