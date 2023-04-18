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

            var maxLen = System.Math.Max(input.CheckedShape.Rank, shape.CheckedShape.Size);
            var outputShape = new Expr[maxLen];
            for (var i = 0; i < maxLen; i++)
            {
                if (maxLen == input.CheckedShape.Rank)
                {
                    outputShape[i] = F.Math.Max(F.Tensors.ShapeOf(input)[i], i < input.CheckedShape.Rank - shape.CheckedShape.Size ? 1 : shape[i]);
                }
                else
                {
                    outputShape[i] = F.Math.Max(i < shape.CheckedShape.Size - input.CheckedShape.Rank ? 1 : F.Tensors.ShapeOf(input)[i], shape[i]);
                }
            }

            return F.Tensors.Expand(input, F.Tensors.Stack(new IR.Tuple(outputShape), 0));
        }
    }
}
