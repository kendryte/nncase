// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform.Mutators;

/// <summary>
/// unroll loop
/// </summary>
internal sealed class UnRollLoop : ExprMutator
{
    /// <inheritdoc/>
    /// shouldn't change the funciton
    public override Expr Visit(Function expr) => expr;

    /// <inheritdoc/>
    public override Expr Visit(TIR.For expr)
    {
        if (!ExpressionMemo.TryGetValue(expr, out var result))
        {
            result = TryUnroll(expr);
            if (!object.ReferenceEquals(result, expr))
            {
                IsMutated = true;
                ExpressionMemo[expr] = result;
                // only unroll one for loop
                return result;
            }
            return base.Visit(expr);
        }
        return result;
    }

    bool IsCanUnroll(TIR.For for_loop)
    {
        return (for_loop.Domain.Start is TensorConst && for_loop.Domain.Stop is TensorConst && for_loop.Domain.Step is TensorConst && for_loop.Mode == LoopMode.Unrolled);
    }

    public static IEnumerable<int> MakeGrid(TIR.For loop)
    {
        int start = ((TensorConst)loop.Domain.Start).Value.ToScalar<int>();
        int stop = ((TensorConst)loop.Domain.Stop).Value.ToScalar<int>();
        int step = ((TensorConst)loop.Domain.Step).Value.ToScalar<int>();

        for (int i = start; i < stop; i += step)
        {
            yield return i;
        }
    }

    Expr TryUnroll(TIR.For expr)
    {
        // try collect the all loops
        var nested_loops = new List<TIR.For>();

        var outter_for = expr;
        if (IsCanUnroll(outter_for))
            nested_loops.Add(outter_for);
        else
            return outter_for;

        while (outter_for.Body.Count == 1 && outter_for.Body[0] is TIR.For inner_for)
        {
            if (IsCanUnroll(inner_for))
                nested_loops.Add(inner_for);
            else
                break;
            outter_for = inner_for;
        }

        var unrolled = (
          from grid in LinqExtensions.CartesianProduct((from loop in nested_loops select MakeGrid(loop)))
          select grid.ToArray()).
          Select(grid =>
            {
                var vmap = new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance);
                for (int i = 0; i < grid.Length; i++)
                {
                    vmap.Add(nested_loops[i].LoopVar, grid[i]);
                }
                return vmap;
            }).
          Select(vmap => new Substitutor(e =>
            {
                if (e is Var v && vmap.TryGetValue(v, out var res))
                    return res;
                return null;
            }).Visit(Visit(expr.Body)));

        return new Sequential(new IRArray<Expr>(unrolled));
    }
}