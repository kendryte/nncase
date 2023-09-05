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
        var i64Axis = Cast(axis, DataTypes.Int64);
        return new If(i64Axis < 0L, i64Axis + rank, i64Axis);
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
        return SliceAndMerge(shapeExpr, index, value, 1L);
    }

    public static Expr Insert(Expr shapeExpr, Expr index, Expr value)
    {
        if (shapeExpr.CheckedShape.IsScalar)
        {
            return SliceAndMerge(StackScalar(shapeExpr), index, value, 0L);
        }

        return SliceAndMerge(shapeExpr, index, value, 0L);
    }

    public static Expr ReplaceList(Expr shapeExpr, Expr list, Expr value)
    {
        return SliceAndMerge(shapeExpr, list, value, 1L, false);
    }

    public static Expr Remove(Expr shapeExpr, Expr index)
    {
        var front = Slice(shapeExpr, 0, index);
        var last = Slice(shapeExpr, index + 1L, int.MaxValue);
        return Concat(new IR.Tuple(front, last), 0);
    }

    public static Expr StackOne(Expr expr)
    {
        return Stack(new IR.Tuple(expr), 0);
    }

    private static Expr SliceAndMerge(Expr originShapeExpr, Expr index, Expr value, Expr indexOffset, bool valueIsList = true)
    {
        var shapeExpr = Cast(originShapeExpr, DataTypes.Int64);
        var front = Slice(shapeExpr, 0, index);
        var last = Slice(shapeExpr, Cast(index, DataTypes.Int64) + indexOffset, int.MaxValue);
        var c = valueIsList ? StackOne(value) : value;
        if (c.CheckedShape.IsScalar)
        {
            c = StackOne(c);
        }

        return Concat(new IR.Tuple(front, c, last), 0);
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

    public static IValue GetShapeValue(Call call)
    {
        call.InferenceType();
        return Value.FromTensor(call.CheckedShape.ToValueArray().Select(x => (long)x).ToArray());
    }
}
