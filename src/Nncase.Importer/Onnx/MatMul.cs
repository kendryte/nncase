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
        private Expr VisitMatMul(in NodeProto op)
        {
            var (a, b) = GetInputExprs(op, 0, 1);
            var shapeA = IR.F.Tensors.ShapeOf(a);
            var shapeB = IR.F.Tensors.ShapeOf(b);
            var lhs = a;
            if (a.CheckedShape.Rank > 2)
            {
                var newShapeA = new Expr[] { -1L, shapeA[-2], shapeA[-1] };
                lhs = IR.F.Tensors.Reshape(a, IR.F.Tensors.Stack(new IR.Tuple(newShapeA), 0));
            }

            if (a.CheckedShape.Rank == 1)
            {
                var newShapeA = new Expr[] { 1L, shapeA[0] };
                lhs = IR.F.Tensors.Reshape(a, IR.F.Tensors.Stack(new IR.Tuple(newShapeA), 0));
            }

            var rhs = b;
            if (b.CheckedShape.Rank > 2)
            {
                var newShapeB = new Expr[] { -1L, shapeB[-2], shapeB[-1] };
                rhs = IR.F.Tensors.Reshape(b, IR.F.Tensors.Stack(new IR.Tuple(newShapeB), 0));
            }

            if (b.CheckedShape.Rank == 1)
            {
                var newShapeB = new Expr[] { shapeB[0], 1L };
                rhs = IR.F.Tensors.Reshape(b, IR.F.Tensors.Stack(new IR.Tuple(newShapeB), 0));
            }

            var maxRank = Math.Max(a.CheckedShape.Rank, b.CheckedShape.Rank);
            var outputShape = new Expr[maxRank];

            if (maxRank == 1)
            {
                outputShape[0] = 1L;
            }
            else if (maxRank == 2)
            {
                if (a.CheckedShape.Rank == 1 && b.CheckedShape.Rank == 2)
                {
                    Array.Resize(ref outputShape, 1);
                    outputShape[0] = shapeB[1];
                }
                else if (a.CheckedShape.Rank == 2 && b.CheckedShape.Rank == 1)
                {
                    Array.Resize(ref outputShape, 1);
                    outputShape[0] = shapeA[0];
                }
                else
                {
                    outputShape[0] = shapeA[0];
                    outputShape[1] = shapeB[1];
                }
            }
            else
            {
                if (maxRank == a.CheckedShape.Rank)
                {
                    if (b.CheckedShape.Rank == 1)
                    {
                        Array.Resize(ref outputShape, maxRank - 1);
                        for (var i = 0; i < maxRank - 2; i++)
                        {
                            outputShape[i] = shapeA[i];
                        }

                        outputShape[^1] = shapeA[-2];
                    }
                    else
                    {
                        for (var i = 0; i < maxRank - 2; i++)
                        {
                            var diff = a.CheckedShape.Rank - b.CheckedShape.Rank;
                            var dimB = i < diff ? 1L : shapeB[i - diff];
                            outputShape[i] = IR.F.Math.Max(shapeA[i], dimB);
                        }

                        outputShape[^2] = shapeA[-2];
                        outputShape[^1] = shapeB[-1];
                    }
                }
                else if (maxRank == b.CheckedShape.Rank)
                {
                    if (a.CheckedShape.Rank == 1)
                    {
                        Array.Resize(ref outputShape, maxRank - 1);
                        for (var i = 0; i < maxRank - 2; i++)
                        {
                            outputShape[i] = shapeB[i];
                        }

                        outputShape[^1] = shapeB[-1];
                    }
                    else
                    {
                        for (var i = 0; i < maxRank - 2; i++)
                        {
                            var diff = b.CheckedShape.Rank - a.CheckedShape.Rank;
                            var dimA = i < diff ? 1L : shapeA[i - diff];
                            outputShape[i] = IR.F.Math.Max(shapeB[i], dimA);
                        }

                        outputShape[^2] = shapeA[-2];
                        outputShape[^1] = shapeB[-1];
                    }
                }
            }

            return IR.F.Tensors.Reshape(F.Tensors.MatMul(lhs, rhs), IR.F.Tensors.Stack(new IR.Tuple(outputShape), 0));
        }
    }
}
