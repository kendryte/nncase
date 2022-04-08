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
internal sealed class UnrollLoop : ExprMutator
{
    /// <inheritdoc/>
    public override Expr MutateLeaf(TIR.For expr)
    {
        return unroll(expr);
    }

    Expr unroll(TIR.For expr)
    {
        if (expr.Dom.Start is not TensorConst || expr.Dom.Stop is not TensorConst || expr.Dom.Step is not TensorConst)
            return expr;
        if (expr.Mode != LoopMode.Unrolled)
            return expr;
        Dictionary<Var, Expr> vmap = new(ReferenceEqualityComparer.Instance);
        List<Expr> unrolled = new();
        for (int i = ((TensorConst)expr.Dom.Start).Value.ToScalar<int>();
                  i < ((TensorConst)expr.Dom.Stop).Value.ToScalar<int>();
                  i += ((TensorConst)expr.Dom.Step).Value.ToScalar<int>())
        {
            vmap[expr.LoopVar] = i;
            Expr step = new Substitutor(e =>
            {
                if (e is Var v && vmap.ContainsKey(v))
                    return vmap[v];
                return e;
            }).Visit(Visit(expr.Body));

            unrolled.Add(step);
        }
        return new Sequential(unrolled);
    }
}