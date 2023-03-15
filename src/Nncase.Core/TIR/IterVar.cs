// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
///  Iteration Variable like a symobl, It represents an iteration over an integer interval.
/// </summary>
public sealed class IterVar : Expr
{
    public IterVar(Range dom, IterationMode mode, Var value)
        : base(new Expr[] { dom, value })
    {
        Mode = mode;
    }

    /// <summary>
    /// Gets the domain of iteration, if known, can be None For the intermediate schedule node, before schedule.
    /// </summary>
    public Range Dom => (Range)Operands[0];

    /// <summary>
    /// Gets the type of the IterVar.
    /// </summary>
    public IterationMode Mode { get; }

    /// <summary>
    /// Gets the looping variable.
    /// </summary>
    public Var Value => (Var)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitIterVar(this, context);

    public IterVar With(Range? dom = null, IterationMode? mode = null, Var? value = null)
        => new(dom ?? Dom, mode ?? Mode, value ?? Value);
}
