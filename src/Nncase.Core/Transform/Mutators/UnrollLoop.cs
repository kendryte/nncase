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
    private readonly HashSet<For> _candidates = new(ReferenceEqualityComparer.Instance);
    private bool foldAll;
    public UnRollLoop(params For[] for_loops)
    {
        foreach (var f in for_loops)
        {
            _candidates.Add(f);
        }
        foldAll = _candidates.Count == 0;
    }

    /// <inheritdoc/>
    public override Expr MutateLeaf(TIR.For expr)
    {
        return foldAll ? unroll(expr) : _candidates.Contains(expr) ? unroll(expr) : expr;
    }

    Expr unroll(TIR.For expr)
    {
        if (expr.Domain.Start is not TensorConst || expr.Domain.Stop is not TensorConst || expr.Domain.Step is not TensorConst)
            return expr;
        if (expr.Mode != LoopMode.Unrolled)
            return expr;
        Dictionary<Var, Expr> vmap = new(ReferenceEqualityComparer.Instance);
        List<Expr> unrolled = new();
        int start = ((TensorConst)expr.Domain.Start).Value.ToScalar<int>();
        int stop = ((TensorConst)expr.Domain.Stop).Value.ToScalar<int>();
        int step = ((TensorConst)expr.Domain.Step).Value.ToScalar<int>();
        for (int i = start; i < stop; i += step)
        {
            vmap[expr.LoopVar] = i;

            Expr body = new Substitutor(e =>
            {
                if (e is Var v && vmap.ContainsKey(v))
                    return vmap[v];
                return e;
            }).Visit(Visit(expr.Body));

            unrolled.Add(body);
        }
        return new Sequential(new IRArray<Expr>(unrolled));
    }
}