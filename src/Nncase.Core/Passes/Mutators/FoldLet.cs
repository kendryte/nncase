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
/// unroll loop.
/// </summary>
public sealed class FoldLet : ExprRewriter
{
    /// <inheritdoc/>
    protected internal override Expr VisitLet(Let expr, Unit context)
    {
        if (HasVisited(expr, out var result))
        {
            return result;
        }

        var replace = (Let)base.VisitLet(expr, context);
        if (replace.Expression is Const @const)
        {
            ProcessScopedRewrite(replace.Var, @const, new HashSet<Expr>(ExprCollector.Collect(expr.Body)));
        }

        return replace;
    }
}
