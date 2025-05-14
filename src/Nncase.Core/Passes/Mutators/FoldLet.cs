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
    protected override BaseExpr RewriteLeafLet(Let expr)
    {
        if (expr.Expression is Const or DimConst or RankedShape { IsFixed: true })
        {
            return new SubFieldRewriter(expr.Var, expr.Expression).Rewrite(expr.Body);
        }

        return expr;
    }

    private sealed class SubFieldRewriter : ExprRewriter
    {
        private readonly IVar _var;
        private readonly BaseExpr _const;

        public SubFieldRewriter(IVar @var, BaseExpr @const)
        {
            _var = @var;
            _const = @const;
        }

        protected override BaseExpr RewriteLeafVar(Var expr)
        {
            if (ReferenceEquals(expr, _var))
            {
                return _const;
            }

            return expr;
        }

        protected override BaseExpr RewriteLeafDimVar(DimVar expr)
        {
            if (ReferenceEquals(expr, _var))
            {
                return _const;
            }

            return expr;
        }

        protected override BaseExpr RewriteLeafShapeVar(ShapeVar expr)
        {
            if (ReferenceEquals(expr, _var))
            {
                return _const;
            }

            return expr;
        }
    }
}
