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
        if (!(expr.Domain.Start is DimConst start &&
            expr.Domain.Stop is DimConst stop &&
            expr.Domain.Step is DimConst step))
        {
            return expr;
        }

        long startv = start.Value;
        long stopv = stop.Value;
        long stepv = step.Value;

        var extent = stopv - startv;
        var (_, rem) = Math.DivRem(extent, stepv);
        if (rem == 0)
        {
            return expr;
        }

        Dictionary<Type, Evaluator.IEvaluator> evaluator_cache = new();
        Dictionary<BaseExpr, BaseExpr> cseMemo = new();
        var vmaps = new Dictionary<IVar, long>(ReferenceEqualityComparer.Instance) { { expr.LoopVar, stopv - rem } };
        var tailBody = new LoopBodyCloner(vmaps, evaluator_cache, cseMemo).Clone(expr.Body, default);
        Expr mainBody = (stopv - rem == startv) ? T.Nop() : new TIR.For(expr.LoopVar, new TIR.Range(expr.Domain.Start, expr.Domain.Stop - rem, expr.Domain.Step), expr.Mode, expr.Body);
        return T.Sequential(mainBody, tailBody);
    }
}
