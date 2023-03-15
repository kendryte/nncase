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
/// A for loop, with poissible type annotations.
/// <example>
/// <code>
///   for (loop_var = min; loop_var &lt; min + extent; ++loop_var) {
///     body
///    }
/// </code>
/// </example>
/// </summary>
public sealed class For : Expr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="For"/> class.
    /// </summary>
    /// <param name="loopVar">The loop variable.</param>
    /// <param name="domain">The domain of for range.</param>
    /// <param name="mode">The kind of the for loop.</param>
    /// <param name="body">The body sequence.</param>
    public For(Var loopVar, Range domain, LoopMode mode, Sequential body)
        : base(new Expr[] { loopVar, domain, body })
    {
        Mode = mode;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="For"/> class.
    /// </summary>
    /// <param name="loopVar">The loop variable.</param>
    /// <param name="domain">The domain of for range.</param>
    /// <param name="mode">The kind of the for loop.</param>
    public For(Var loopVar, Range domain, LoopMode mode)
        : this(loopVar, domain, mode, new())
    {
    }

    /// <summary>
    /// Gets the loop variable.
    /// </summary>
    public Var LoopVar => (Var)Operands[0];

    /// <summary>
    /// Gets the domain of for range.
    /// </summary>
    public Range Domain => (Range)Operands[1];

    /// <summary>
    /// Gets the kind of the for loop.
    /// </summary>
    public LoopMode Mode { get; }

    /// <summary>
    /// Gets the body sequence.
    /// </summary>
    public Sequential Body => (Sequential)Operands[2];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitFor(this, context);

    public For With(Var? loopVar = null, Range? domain = null, LoopMode? loopMode = null, Sequential? body = null)
        => new For(loopVar ?? LoopVar, domain ?? Domain, loopMode ?? Mode, body ?? Body);
}
