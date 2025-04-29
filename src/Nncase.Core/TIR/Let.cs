// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// Let binding. Bind var to value then evaluate body. return unit.
/// </summary>
public sealed class Let : Expr
{
    public Let(IVar var, BaseExpr expression, Sequential body)
        : base([(BaseExpr)var, expression, body])
    {
    }

    /// <summary>
    /// Gets the expr.
    /// </summary>
    public IVar Var => (IVar)Operands[0];

    /// <summary>
    /// Gets the value to be binded.
    /// </summary>
    public BaseExpr Expression => Operands[1];

    /// <summary>
    /// Gets the Let body.
    /// </summary>
    public Sequential Body => (Sequential)Operands[2];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitLet(this, context);

    public Let With(IVar? var = null, BaseExpr? expression = null, Sequential? body = null)
        => new Let(var ?? Var, expression ?? Expression, body ?? Body);
}
