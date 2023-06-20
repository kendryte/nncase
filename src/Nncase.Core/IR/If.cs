// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// if(Condition) then { Then } else { Else }.
/// </summary>
public sealed class If : Expr
{
    public If(Expr condition, Expr then, Expr @else, params Expr[] paramList)
        : base(paramList.Concat(new[] { condition, then, @else }))
    {
    }

    public Expr Condition => Operands[^3];

    public Expr Then => Operands[^2];

    public Expr Else => Operands[^1];

    public Expr[] ParamList => Operands[..^3].ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitIf(this, context);

    public If With(Expr? condition = null, Expr? then = null, Expr? @else = null, Expr[]? paramList = null)
        => new If(condition ?? Condition, then ?? Then, @else ?? Else, paramList ?? ParamList);
}
