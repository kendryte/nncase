// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule;

internal static class OrToolsExtensions
{
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
}
