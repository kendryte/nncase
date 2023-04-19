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

            if (shape is TensorConst)
            {
                var maxLen = System.Math.Max(input.CheckedShape.Rank, shape.CheckedShape.Size);
                var outputShape = new Expr[maxLen];
                var inputShape = F.Tensors.ShapeOf(input);
                for (var i = 0; i < maxLen; i++)
                {
                    if (maxLen == input.CheckedShape.Rank)
                    {
                        outputShape[i] = F.Math.Max(inputShape[i], i < input.CheckedShape.Rank - shape.CheckedShape.Size ? 1L : shape[i - (input.CheckedShape.Rank - shape.CheckedShape.Size)]);
                    }
                    else
                    {
                        outputShape[i] = F.Math.Max(i < shape.CheckedShape.Size - input.CheckedShape.Rank ? 1L : inputShape[i - (shape.CheckedShape.Size - input.CheckedShape.Rank)], shape[i]);
                    }
                }

                return F.Tensors.Expand(input, F.Tensors.Stack(new IR.Tuple(outputShape), 0));
            }

            return F.Tensors.Expand(input, shape);
        }
    }
}
