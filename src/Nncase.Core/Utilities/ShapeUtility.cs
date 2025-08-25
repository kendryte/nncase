// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using static Nncase.IR.F.Tensors;
using Isl = IntegerSetLibrary;

namespace Nncase.Utilities;

public static class ShapeUtility
{
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

    public static Shape GetPermutation(Expr tensor, long[] dims)
    {
        var r = tensor.CheckedShape.Rank;

        var fullDims = new List<long>();

        for (int i = 0; i != r; i++)
        {
            fullDims.Add(i);
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
            return fullDims.ToArray();
        }
        else
        {
            throw new NotImplementedException("GetPermuation in Transpose need 2D perm");
        }
    }

    public static ValueRange<double> GetDimSegmentRange(ValueRange<double> range, int segmentIndex, int segmentsCount)
    {
        if (segmentIndex == 0)
        {
            return new(range.Min, range.Min);
        }

        var segmentSize = (int)System.Math.Ceiling((range.Max - range.Min) / (segmentsCount - 1));
        var segmentStart = System.Math.Min(((segmentIndex - 1) * segmentSize) + range.Min + 1, range.Max);
        var segmentEnd = System.Math.Min(segmentStart + segmentSize, range.Max);

        return new(segmentStart, segmentEnd);
    }
}
