// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule;

internal static class OrToolsExtensions
{
    private static readonly Regex _rangePattern = new Regex(@"\(\d+..\d+\)", RegexOptions.Compiled);

    public static IntExpr CeilDiv(this IntExpr numer, long denom) =>
        (numer + (denom - 1)) / denom;

    public static IntExpr CeilDiv(this IntExpr numer, IntExpr denom) =>
        denom.solver().MakeDiv(numer + (denom - 1), denom);

    public static IntExpr CeilDiv(this long numer, IntExpr denom) =>
        denom.solver().MakeDiv(numer + (denom - 1), denom);

    public static IntExpr CeilDiv(this int numer, IntExpr denom) =>
        denom.solver().MakeDiv(numer + (denom - 1), denom);

    public static IntExpr MakeProd(this Solver solver, IntVarVector ints)
    {
        return ints.Skip(1).Aggregate((IntExpr)ints.First(), solver.MakeProd);
    }

    public static long[][] Value(this Assignment sol, IntExpr[][] inputs)
    {
        var mat = new long[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
        {
            mat[i] = new long[inputs[i].Length];
            for (int j = 0; j < inputs[i].Length; j++)
            {
                mat[i][j] = sol.Value(inputs[i][j].Var());
            }
        }

        return mat;
    }

    public static long[] Value(this Assignment sol, IntExpr[] inputs)
    {
        var vec = new long[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            vec[i] = sol.Value(inputs[i].Var());
        }

        return vec;
    }

    public static string ToSimplifyString(this PropagationBaseObject intExpr)
    {
        var str = intExpr.ToString();
        return _rangePattern.Replace(str, string.Empty);
    }
}
