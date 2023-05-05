// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitReshape(in NodeProto op)
        {
            var (input, shape) = GetInputExprs(op, 0, 1);
            var inputShape = F.Tensors.ShapeOf(input);
            if (shape is TensorConst)
            {
                var shapeValue = ((TensorConst)shape).Value.ToArray<long>();
                var actualShape = new Expr[shapeValue.Length];
                var negAxis = shapeValue.Length;
                for (int i = 0; i < actualShape.Length; i++)
                {
                    if (shapeValue[i] == 0L)
                    {
                        actualShape[i] = inputShape[i];
                    }
                    else if (shapeValue[i] == -1L)
                    {
                        negAxis = i;
                    }
                    else
                    {
                        actualShape[i] = shapeValue[i];
                    }
                }

                if (negAxis < shapeValue.Length)
                {
                    Expr productOut = 1L;
                    for (int i = 0; i < shapeValue.Length; i++)
                    {
                        if (i != negAxis)
                        {
                            productOut *= actualShape[i];
                        }
                    }

                    Expr productIn = F.Tensors.Prod(inputShape);

                    actualShape[negAxis] = productIn / productOut;
                }

                return F.Tensors.Reshape(input, F.Tensors.Stack(new IR.Tuple(actualShape), 0));
            }
            else
            {
                // allowzero has been avaliable since opset 14
                var allowZero = GetBoolAttribute(op, "allowzero", false);
                if (allowZero)
                {
                    throw new NotSupportedException("Not support reshape attribute: allowzero");
                }

                return F.Tensors.Reshape(input, shape);
            }
        }
    }
}
