// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Utilities;

namespace Nncase.IR;

/// <summary>
/// if(Condition) then { Then } else { Else }.
/// </summary>
public sealed class If : BaseCall
{
    public If(Expr condition, BaseFunction then, BaseFunction @else, ReadOnlySpan<Expr> arguments)
        : base(ArrayUtility.Concat(condition, then, @else, arguments))
    {
    }

    public If(Expr condition, BaseFunction then, BaseFunction @else, params Expr[] arguments)
        : this(condition, then, @else, (ReadOnlySpan<Expr>)arguments)
    {
    }

    public Expr Condition => Operands[0];

    public BaseFunction Then => (BaseFunction)Operands[1];

    public BaseFunction Else => (BaseFunction)Operands[2];

    public override ReadOnlySpan<Expr> Arguments => Operands[3..];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitIf(this, context);

    public If With(Expr? condition = null, BaseFunction? then = null, BaseFunction? @else = null, Expr[]? arguments = null)
        => new If(condition ?? Condition, then ?? Then, @else ?? Else, arguments ?? Arguments);
}
