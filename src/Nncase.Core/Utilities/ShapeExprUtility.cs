// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using GiGraph.Dot.Output.Writers.Edges;
using Nncase.IR;
using Nncase.IR.Tensors;
using static Nncase.IR.F.Tensors;

namespace Nncase.Utilities;

public static class ShapeExprUtility
{
    public static Expr BroadcastShape(Expr lhsShape, params Expr[] rhsShape)
    {
        var tmpTensor = new[] { ConstantOfShape(lhsShape, 0) }
            .Concat(rhsShape)
            .Aggregate((sum, shape) => ConstantOfShape(shape, 0) * sum);
        return ShapeOf(tmpTensor);
    }

    public static Expr Positive(Expr axis, Expr inShape)
    {
        var rank = ShapeOf(inShape)[0];
        var i32Axis = Cast(axis, DataTypes.Int32);
        return new If(i32Axis < 0, i32Axis + rank, i32Axis);
    }

    private static Expr CheckShape(Expr shape)
    {
        if (shape.CheckedType == null)
        {
            shape.InferenceType();
        }

        if (shape.CheckedType is not TensorType || shape.CheckedShape.Count == 0)
        {
            // CompilerServices.DumpIR(shape, "checkShape", "/Users/homura/Code/nncase-fix/tests_output/ShapeBucketTest/TestModel/ShapeExpr/");
            throw new InvalidOperationException();
        }

        return shape;
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

    // public static Expr ShapeOf(Expr expr) => expr.EvaluateShapeExpr();
    public static Expr ShapeOf(Expr expr) => Cast(IR.F.Tensors.ShapeOf(expr), DataTypes.Int32);

    private static Expr SliceAndMerge(Expr shapeExpr, Expr index, Expr value, Expr indexOffset, bool valueIsList = true)
    {
        var front = Slice(shapeExpr, 0, index);
        var last = Slice(shapeExpr, Cast(index, DataTypes.Int32) + indexOffset, int.MaxValue);
        return Concat(new IR.Tuple(front, valueIsList ? StackOne(value) : value, last), 0);
    }

    private static Expr StackOne(Expr expr)
    {
        // return new If(Rank(expr) > 0L, expr, Stack(new IR.Tuple(expr), 0));
        return Stack(new IR.Tuple(expr), 0);
    }
}
