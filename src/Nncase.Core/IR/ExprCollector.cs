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

public sealed class ExprCollector : ExprWalker<List<BaseExpr>>
{
    private ExprCollector()
    {
    }

    public static IReadOnlyList<BaseExpr> Collect(BaseExpr expr)
    {
        var exprs = new List<BaseExpr>();
        new ExprCollector().Visit(expr, exprs);
        return exprs;
    }

    protected override Unit DefaultVisitLeaf(BaseExpr expr, List<BaseExpr> context)
    {
        context.Add(expr);
        return default;
    }
}
