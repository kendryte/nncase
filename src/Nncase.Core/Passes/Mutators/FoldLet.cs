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
/// fold let.
/// </summary>
public sealed class FoldLet : ExprRewriter
{
    /// <inheritdoc/>
    protected override Expr RewriteLeafLet(Let expr)
    {
        if (expr.Expression is Const @const)
        {
            return new SubFieldRewriter(expr.Var, @const).Rewrite(expr.Body);
        }

        return expr;
    }

    private sealed class SubFieldRewriter : ExprRewriter
    {
        private readonly Var _var;
        private readonly Const _const;

        public SubFieldRewriter(Var @var, Const @const)
        {
            _var = @var;
            _const = @const;
        }

        protected override Expr RewriteLeafVar(Var expr)
        {
            if (ReferenceEquals(expr, _var))
            {
                return _const;
            }

            return expr;
        }
    }
}
