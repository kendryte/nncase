// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR;

namespace Nncase.Passes.Tile;

internal sealed class PrimTileVisitor : ExprVisitor<Unit, Unit>
{
    public PrimTileVisitor()
    {
        TileList = new();
        NameList = new();
        Count = 0;
    }

    public List<KeyValuePair<Expr, int[]>> TileList { get; }

    public List<KeyValuePair<Expr, string>> NameList { get; }

    public int Count { get; private set; }

    protected override Unit DefaultVisitLeaf(Expr expr)
    {
        return Unit.Default;
    }

    protected override Unit VisitLeafCall(Call expr)
    {
        switch (expr.Target)
        {
            case IR.Math.MatMul op:
                {
                    var lhs = expr.Arguments[0];
                    var rhs = expr.Arguments[1];
                    var inTileShapeA = Enumerable.Repeat(1, lhs.CheckedShape.Rank).ToArray();
                    Array.Fill(inTileShapeA, 32, inTileShapeA.Length - 2, 2);
                    var inTileShapeB = Enumerable.Repeat(1, rhs.CheckedShape.Rank).ToArray();
                    Array.Fill(inTileShapeB, 32, inTileShapeB.Length - 2, 2);

                    if (!(lhs is Var or TensorConst))
                    {
                        var oldTileAShape = TileList.Find(k => k.Key == lhs).Value;
                        inTileShapeA = inTileShapeA.Select((s, i) => Math.Max(s, oldTileAShape[i])).ToArray();
                    }
                    else
                    {
                        TileList.Add(new(lhs, inTileShapeA));
                        NameList.Add(new(lhs, nameof(IR.Math.MatMul) + "_" + Count.ToString() + "_lhs"));
                    }

                    if (!(rhs is Var or TensorConst))
                    {
                        var oldTileBShape = TileList.Find(k => k.Key == rhs).Value;
                        inTileShapeB = inTileShapeB.Select((s, i) => Math.Max(s, oldTileBShape[i])).ToArray();
                    }
                    else
                    {
                        TileList.Add(new(rhs, inTileShapeB));
                        NameList.Add(new(rhs, nameof(IR.Math.MatMul) + "_" + Count.ToString() + "_rhs"));
                    }

                    var outTileShape = Enumerable.Repeat(1, expr.CheckedShape.Rank).ToArray();
                    outTileShape[^1] = inTileShapeB[^1];
                    outTileShape[^2] = inTileShapeA[^2];
                    TileList.Add(new(expr, outTileShape));
                    NameList.Add(new(expr, nameof(IR.Math.MatMul) + "_" + Count.ToString()));
                    Count++;
                    break;
                }

            case IR.Math.Unary or IR.CPU.Store or IR.CPU.Load:
                {
                    var input = expr.Arguments[0];
                    var inTileShape = Enumerable.Repeat(1, input.CheckedShape.Rank).ToArray();
                    inTileShape[^1] = 32;

                    if (!(input is Var or TensorConst))
                    {
                        var oldTileShape = TileList.Find(k => k.Key == input).Value;
                        inTileShape = inTileShape.Select((s, i) => Math.Max(s, oldTileShape[i])).ToArray();
                    }
                    else
                    {
                        TileList.Add(new(input, inTileShape));
                        NameList.Add(new(expr, expr.Target.GetType().Name + "_" + Count.ToString() + "_input"));
                    }

                    var outTileShape = inTileShape;
                    TileList.Add(new(expr, outTileShape));
                    NameList.Add(new(expr, expr.Target.GetType().Name + "_" + Count.ToString()));
                    Count++;
                    break;
                }

            case IR.Math.Binary op:
                {
                    var lhs = expr.Arguments[0];
                    var rhs = expr.Arguments[1];
                    var inTileShapeA = Enumerable.Repeat(1, lhs.CheckedShape.Rank).ToArray();
                    inTileShapeA[^1] = 32;
                    var inTileShapeB = Enumerable.Repeat(1, rhs.CheckedShape.Rank).ToArray();
                    inTileShapeB[^1] = 32;

                    if (!(lhs is Var or TensorConst))
                    {
                        var oldTileAShape = TileList.Find(k => k.Key == lhs).Value;
                        inTileShapeA = inTileShapeA.Select((s, i) => Math.Max(s, oldTileAShape[i])).ToArray();
                    }
                    else
                    {
                        TileList.Add(new(lhs, inTileShapeA));
                        NameList.Add(new(lhs, nameof(IR.Math.Binary) + "_" + Count + "_lhs"));
                    }

                    if (!(rhs is Var or TensorConst))
                    {
                        var oldTileBShape = TileList.Find(k => k.Key == rhs).Value;
                        inTileShapeB = inTileShapeB.Select((s, i) => Math.Max(s, oldTileBShape[i])).ToArray();
                    }
                    else
                    {
                        TileList.Add(new(rhs, inTileShapeB));
                        NameList.Add(new(rhs, nameof(IR.Math.Binary) + "_" + Count + "_rhs"));
                    }

                    var outTileShape = Enumerable.Repeat(1, expr.CheckedShape.Rank).ToArray();
                    outTileShape[^1] = 32;
                    TileList.Add(new(expr, outTileShape));
                    NameList.Add(new(expr, nameof(IR.Math.Binary) + "_" + Count));
                    Count++;
                    break;
                }

            default:
                throw new NotImplementedException("Not Implemented Op: " + expr.Target);
        }

        return Unit.Default;
    }
}
