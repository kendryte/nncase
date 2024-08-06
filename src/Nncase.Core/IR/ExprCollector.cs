// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Nncase.TIR;

namespace Nncase.IR;

public sealed class ExprCollector : ExprWalker<List<Expr>>
{
    private readonly Func<Expr, bool>? _condition;

    private ExprCollector(Func<Expr, bool>? condition = null)
    {
        _condition = condition;
    }

    public static IReadOnlyList<Expr> Collect(Expr expr, Func<Expr, bool>? condition = null)
    {
        var exprs = new List<Expr>();
        new ExprCollector(condition).Visit(expr, exprs);
        return exprs;
    }

    protected override Unit DefaultVisitLeaf(Expr expr, List<Expr> context)
    {
        if (_condition?.Invoke(expr) ?? true)
        {
            context.Add(expr);
        }

        return default;
    }
}
