// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Expression rewriter.
/// </summary>
/// <typeparam name="TContext">Rewrite context.</typeparam>
public abstract partial class ExprRewriter<TContext> : ExprVisitor<Expr, IRType, TContext>
{
    private IReadOnlySet<Expr>? _rewriteScope;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExprRewriter{TContext}"/> class.
    /// </summary>
    /// <param name="visitOtherFunctions">Vist other functions.</param>
    public ExprRewriter(bool visitOtherFunctions = false)
        : base(visitOtherFunctions)
    {
    }

    /// <summary>
    /// Gets a value indicating whether expression is mutated.
    /// </summary>
    public bool IsMutated { get; private set; }

    /// <summary>
    /// Rewrite expression.
    /// </summary>
    /// <param name="expr">Expression to rewrite.</param>
    /// <param name="context">Context.</param>
    /// <returns>Rewritten expression.</returns>
    public Expr Rewrite(Expr expr, TContext context) => Visit(expr, context);

    /// <summary>
    /// Rewrite expression in a scope.
    /// </summary>
    /// <param name="expr">Expression to rewrite.</param>
    /// <param name="context">Context.</param>
    /// <param name="scope">Rewrite scope.</param>
    /// <returns>Rewritten expression.</returns>
    public Expr ScopedRewrite(Expr expr, TContext context, IReadOnlySet<Expr>? scope = null)
    {
        _rewriteScope = scope ?? new HashSet<Expr>(ExprCollector.Collect(expr));
        return Rewrite(expr, context);
    }

    /// <summary>
    /// Default rewrite leaf routine.
    /// </summary>
    protected virtual Expr DefaultRewriteLeaf(Expr expr, TContext context) => expr;

    protected void SetMutated() => IsMutated = true;

    protected Expr ProcessRewrite(Expr original, Expr replace)
    {
        if (!ReferenceEquals(original, replace))
        {
            if (_rewriteScope == null)
            {
                original.ReplaceAllUsesWith(replace);
                SetMutated();
            }
            else
            {
                ProcessScopedRewrite(original, replace, _rewriteScope);
            }
        }

        return replace;
    }

    protected void ProcessScopedRewrite(Expr original, Expr replace, IReadOnlySet<Expr> scope)
    {
        if (!ReferenceEquals(original, replace))
        {
            original.ReplaceScopedUsesWith(replace, scope);
            SetMutated();
        }
    }
}

/// <summary>
/// Expression rewriter.
/// </summary>
public abstract partial class ExprRewriter : ExprRewriter<Unit>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ExprRewriter"/> class.
    /// </summary>
    /// <param name="visitOtherFunctions">Vist other functions.</param>
    protected ExprRewriter(bool visitOtherFunctions = false)
        : base(visitOtherFunctions)
    {
    }

    /// <summary>
    /// Rewrite expression.
    /// </summary>
    /// <param name="expr">Expression to rewrite.</param>
    /// <returns>Rewritten expression.</returns>
    public Expr Rewrite(Expr expr) => Rewrite(expr, default);

    /// <summary>
    /// Rewrite expression in a scope.
    /// </summary>
    /// <param name="expr">Expression to rewrite.</param>
    /// <param name="scope">Rewrite scope.</param>
    /// <returns>Rewritten expression.</returns>
    public Expr ScopedRewrite(Expr expr, IReadOnlySet<Expr>? scope = null) => ScopedRewrite(expr, default, scope);

    /// <summary>
    /// Default rewrite leaf routine.
    /// </summary>
    protected virtual Expr DefaultRewriteLeaf(Expr expr) => base.DefaultRewriteLeaf(expr, default);

    /// <inheritdoc/>
    protected sealed override Expr DefaultRewriteLeaf(Expr expr, Unit context) => DefaultRewriteLeaf(expr);
}
