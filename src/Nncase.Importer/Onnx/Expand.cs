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
        private Expr VisitExpand(in NodeProto op)
        {
            var (input, shape) = GetInputExprs(op, 0, 1);

            // TODO: support input.CheckedShape.Rank != shape.CheckedShape.Size
            var outputShape = new Expr[input.CheckedShape.Rank];
            var inputShape = F.Tensors.ShapeOf(input);
            for (var i = 0; i < input.CheckedShape.Rank; i++)
            {
                outputShape[i] = F.Math.Max(inputShape[i], shape[i]);
            }

            return F.Tensors.Expand(input, F.Tensors.Stack(new IR.Tuple(outputShape), 0));
        }
    }
}
