// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

public sealed class TailLoopStripping : ExprRewriter
{
    /// <inheritdoc/>
    protected override Expr RewriteLeafFor(For expr)
    {
        if (!(expr.Domain.Start is TensorConst start &&
            expr.Domain.Stop is TensorConst stop &&
            expr.Domain.Step is TensorConst step))
        {
            return expr;
        }

        int startv = start.Value.ToScalar<int>();
        int stopv = stop.Value.ToScalar<int>();
        int stepv = step.Value.ToScalar<int>();

        var extent = stopv - startv;
        var (div, rem) = Math.DivRem(extent, stepv);
        if (rem == 0)
        {
            return expr;
        }

        Dictionary<Type, Evaluator.IEvaluator> evaluator_cache = new();
        Dictionary<Expr, Expr> cseMemo = new();
        var vmaps = new Dictionary<Var, TensorConst>(ReferenceEqualityComparer.Instance) { { expr.LoopVar, stopv - rem } };
        var tailBody = new LoopBodyCloner(vmaps, evaluator_cache, cseMemo).Clone(expr.Body, default);
        Expr mainBody = (stopv - rem == startv) ? T.Nop() : new TIR.For(expr.LoopVar, new TIR.Range(expr.Domain.Start, expr.Domain.Stop - rem, expr.Domain.Step), expr.Mode, expr.Body);
        return T.Sequential(mainBody, tailBody);
    }
}
