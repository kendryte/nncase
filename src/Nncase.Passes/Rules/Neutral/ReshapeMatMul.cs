// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.MetadataUtility;

namespace Nncase.Passes.Rules.Neutral;

[RuleGenerator]
public partial class ReshapeMatMul : RewriteRule<Pattern>
{
    public override Pattern Pattern => IsMatMul(
        null,
        "matmul",
        IsWildcard("a") with { TypePattern = HasFixedShape() },
        IsWildcard("b") with { TypePattern = HasFixedShape() });

    private Expr? GetReplace(Expr a, Expr b)
    {
        if (a.CheckedShape.Rank > 2 && b.CheckedShape.Rank > 2)
        {
            return null;
        }

        var lhs = a;
        var shapeA = a.CheckedShape.ToValueArray();
        if (a.CheckedShape.Rank > 2)
        {
            var c = shapeA.Take(a.CheckedShape.Rank - 2).Aggregate(1, (sum, x) => x * sum);
            var newShapeA = new long[] { c, shapeA[^2], shapeA[^1] };
            lhs = IR.F.Tensors.Reshape(a, newShapeA);
        }

        if (a.CheckedShape.Rank == 1)
        {
            var newShapeA = new long[] { 1L, shapeA[0] };
            lhs = IR.F.Tensors.Reshape(a, newShapeA);
        }

        var rhs = b;
        var shapeB = b.CheckedShape.ToValueArray();
        if (b.CheckedShape.Rank > 2)
        {
            var c = shapeB.Take(b.CheckedShape.Rank - 2).Aggregate(1, (sum, x) => x * sum);
            var newShapeB = new long[] { c, shapeB[^2], shapeB[^1] };
            rhs = IR.F.Tensors.Reshape(b, newShapeB);
        }

        if (b.CheckedShape.Rank == 1)
        {
            var newShapeB = new long[] { shapeB[0], 1L };
            rhs = IR.F.Tensors.Reshape(b, newShapeB);
        }

        var maxRank = Math.Max(a.CheckedShape.Rank, b.CheckedShape.Rank);
        var outputShape = new long[maxRank];

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

                    outputShape[^1] = shapeA[^2];
                }
                else
                {
                    for (var i = 0; i < maxRank - 2; i++)
                    {
                        var diff = a.CheckedShape.Rank - b.CheckedShape.Rank;
                        var dimB = i < diff ? 1L : shapeB[i - diff];
                        outputShape[i] = Math.Max(shapeA[i], dimB);
                    }

                    outputShape[^2] = shapeA[^2];
                    outputShape[^1] = shapeB[^1];
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

                    outputShape[^1] = shapeB[^1];
                }
                else
                {
                    for (var i = 0; i < maxRank - 2; i++)
                    {
                        var diff = b.CheckedShape.Rank - a.CheckedShape.Rank;
                        var dimA = i < diff ? 1L : shapeA[i - diff];
                        outputShape[i] = Math.Max(shapeB[i], dimA);
                    }

                    outputShape[^2] = shapeA[^2];
                    outputShape[^1] = shapeB[^1];
                }
            }
        }

        return IR.F.Tensors.Reshape(IR.F.Tensors.MatMul(lhs, rhs), outputShape);
    }
}
