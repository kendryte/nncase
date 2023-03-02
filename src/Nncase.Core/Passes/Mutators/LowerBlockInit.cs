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

/// <summary>
/// Flatten the multi-dimensional BufferLoad and BufferStore to single dimensional Load/Store. Also remove Block to ensure that the flattened TIR can not be scheduled again.
/// </summary>
public sealed class LowerBlockInit : ExprRewriter
{
    /// <inheritdoc/>
    protected override Expr RewriteLeafBlock(Block expr)
    {
        if (expr.InitBody.Count == 0)
        {
            return expr;
        }

        var initbody = Lowering(expr.InitBody, expr.IterVars);
        return expr.With(
            initBody: Sequential.Empty,
            body: new Sequential(initbody, expr.Body));
    }

    private Expr Lowering(Sequential init, ReadOnlySpan<IterVar> iterVars)
    {
        List<Expr> conds = new();
        foreach (var iterVar in iterVars)
        {
            if (iterVar.Mode == IterationMode.CommReduce)
            {
                conds.Add(IR.F.Math.Equal(iterVar, iterVar.Dom.Start));
            }
        }

        if (conds.Count == 0)
        {
            return init;
        }

        var cond = conds[0];
        foreach (var i in Enumerable.Range(1, conds.Count - 1))
        {
            cond = IR.F.Math.LogicalAnd(cond, conds[i]);
        }

        return new IfThenElse(cond, init);
    }
}
