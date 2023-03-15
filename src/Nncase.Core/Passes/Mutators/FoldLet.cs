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
        if (expr.Expression is Const @const)
        {
            ExprMemo.Add(expr.Var, @const);
            return VisitSequential(expr.Body, context);
        }

        return base.VisitLet(expr, context);
    }
}
