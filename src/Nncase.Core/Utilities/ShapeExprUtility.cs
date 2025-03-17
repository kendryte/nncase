// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using static Nncase.IR.F.Tensors;

namespace Nncase.Utilities;

public static class ShapeExprUtility
{
    public static Expr Positive(Expr axis, Expr inShape)
    {
        var rank = new Call(new Rank(), inShape);
        var i64Axis = Cast(axis, DataTypes.Int64);
        var i64AxisVar1 = new Var(typeAnnotation: DataTypes.Int64);
        var then = new Function(i64AxisVar1 + rank, i64AxisVar1);
        var i64AxisVar2 = new Var(typeAnnotation: DataTypes.Int64);
        var @else = new Function(i64AxisVar2, i64AxisVar2);
        return new If(i64Axis < 0L, then, @else, i64Axis);
    }

    public static Expr If(Expr condition, Func<Expr> thenExpr, Func<Expr> elseExpr)
    {
        var thenFunc = new Function(thenExpr());
        var elseFunc = new Function(elseExpr());
        return new If(condition, thenFunc, elseFunc);
    }

    public static Expr If(Expr condition, Func<Var, Expr> thenExpr, Func<Var, Expr> elseExpr, Expr arg)
    {
        var var1 = new Var(arg.CheckedType);
        var var2 = var1.With();
        var thenFunc = new Function(thenExpr(var1), var1);
        var elseFunc = new Function(elseExpr(var2), var2);
        return new If(condition, thenFunc, elseFunc, arg);
    }

    public static Expr If(Expr condition, Func<Var, Var, Expr> thenExpr, Func<Var, Var, Expr> elseExpr, Expr arg1, Expr arg2)
    {
        var var11 = new Var(arg1.CheckedType);
        var var21 = var11.With();
        var var12 = new Var(arg2.CheckedType);
        var var22 = var12.With();
        var thenFunc = new Function(thenExpr(var11, var12), var11, var12);
        var elseFunc = new Function(elseExpr(var21, var22), var21, var22);
        return new If(condition, thenFunc, elseFunc, arg1, arg2);
    }

    public static Expr If(Expr condition, Func<Var, Var, Var, Expr> thenExpr, Func<Var, Var, Var, Expr> elseExpr, Expr arg1, Expr arg2, Expr arg3)
    {
        var var11 = new Var(arg1.CheckedType);
        var var21 = var11.With();
        var var12 = new Var(arg2.CheckedType);
        var var22 = var12.With();
        var var13 = new Var(arg3.CheckedType);
        var var23 = var13.With();
        var thenFunc = new Function(thenExpr(var11, var12, var13), var11, var12, var13);
        var elseFunc = new Function(elseExpr(var21, var22, var23), var21, var22, var23);
        return new If(condition, thenFunc, elseFunc, arg1, arg2, arg3);
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

    public static IValue GetShapeValue(Call call)
    {
        call.InferenceType();
        return Value.FromTensor(call.CheckedShape.ToValueArray().Select(x => (long)x).ToArray());
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

    public static Expr GetPermutation(Expr tensor, long[] dims)
    {

        var r = tensor.CheckedShape.Rank;

        // System.Console.WriteLine($"shape:{tensor.CheckedShape} rank:{r} dims:{dims[0]},{dims[1]}");
        // format dims to non-negative
        // var newDims = dims.Select(x => x < 0 ? x + r : x).ToArray();
        List<long> fullDims = new List<long>();

        for (int i = 0; i!= r; i++)
        {
            fullDims.Add((long)i);
        }
        for (int i = 0; i != dims.Length; i++)
        {
            if (dims[i] < 0)
            {
                dims[i] = r + dims[i];
            }
        }

        if (dims.Length == 2)
        {
            (fullDims[(int)dims[0]], fullDims[(int)dims[1]]) = (fullDims[(int)dims[1]], fullDims[(int)dims[0]]);
            return Tensor.FromArray(fullDims.ToArray());
        }
        else
        {
            throw new NotImplementedException("GetPermuation in Transpose need 2D perm");
        }
    }
}
