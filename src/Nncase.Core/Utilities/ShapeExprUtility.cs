// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using GiGraph.Dot.Output.Writers.Edges;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using static Nncase.IR.F.Tensors;

namespace Nncase.Utilities;

public static class ShapeExprUtility
{
    public static Expr BroadcastShape(Expr lhsShape, params Expr[] rhsShape)
    {
        return IR.F.ShapeExpr.BroadcastShape(new[] { lhsShape }.Concat(rhsShape).ToArray());
    }

    public static Expr Positive(Expr axis, Expr inShape)
    {
        var rank = new Call(new Rank(), inShape);
        var i32Axis = Cast(axis, DataTypes.Int32);
        return new If(i32Axis < 0, i32Axis + rank, i32Axis);
    }

    public static Expr Slice(Expr shape, int begin, int end)
    {
        return IR.F.Tensors.Slice(CheckShape(shape), new[] { begin }, new[] { end }, 1);
    }

    public static Expr Slice(Expr shape, Expr begin, Expr end)
    {
        return IR.F.Tensors.Slice(CheckShape(shape), StackOne(begin), StackOne(end), 1);
    }

    public static Expr Replace(Expr shapeExpr, Expr index, Expr value)
    {
        return SliceAndMerge(shapeExpr, index, value, 1);
    }

    public static Expr Insert(Expr shapeExpr, Expr index, Expr value)
    {
        return SliceAndMerge(shapeExpr, index, value, 0);
    }

    public static Expr ReplaceList(Expr shapeExpr, Expr list, Expr value)
    {
        return SliceAndMerge(shapeExpr, list, value, 1, false);
    }

    public static Expr Remove(Expr shapeExpr, Expr index)
    {
        var front = Slice(shapeExpr, 0, index);
        var last = Slice(shapeExpr, index + 1, int.MaxValue);
        return Concat(new IR.Tuple(front, last), 0);
    }

    public static Expr RankOf(Expr expr)
    {
        if (!expr.InferenceType())
        {
            DumpScope.Current.DumpIR(expr, "BroadcastShape");
            throw new NotImplementedException();
        }

        return new Call(new Rank(), expr);
    }

    public static Expr StackOne(Expr expr)
    {
        return Stack(new IR.Tuple(expr), 0);
    }

    private static Expr SliceAndMerge(Expr shapeExpr, Expr index, Expr value, Expr indexOffset, bool valueIsList = true)
    {
        var front = Slice(shapeExpr, 0, index);
        var last = Slice(shapeExpr, Cast(index, DataTypes.Int32) + indexOffset, int.MaxValue);
        return Concat(new IR.Tuple(front, valueIsList ? StackOne(value) : value, last), 0);
    }

    private static Expr CheckShape(Expr shape)
    {
        if (shape.CheckedType == null)
        {
            shape.InferenceType();
        }

        if (shape.CheckedType is not TensorType || shape.CheckedShape.IsScalar)
        {
            DumpScope.Current.DumpIR(shape, "ShapeExprUtilityCheckShape");
            throw new InvalidOperationException();
        }

        return shape;
    }
}
