// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.IR.Affine;

/// <summary>
/// Expression visitor.
/// </summary>
/// <typeparam name="TExprResult">Expression visit result type.</typeparam>
/// <typeparam name="TContext">Visit context.</typeparam>
public abstract partial class AffineExprVisitor<TExprResult, TContext>
{
    /// <summary>
    /// Gets expression memo.
    /// </summary>
    public Dictionary<AffineExpr, TExprResult> ExprMemo { get; } = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    /// Gets visit root.
    /// </summary>
    protected AffineExpr? VisitRoot { get; private set; }

    /// <summary>
    /// Visit <see cref="Expr"/>.
    /// </summary>
    public TExprResult Visit(AffineExpr expr, TContext context)
    {
        VisitRoot ??= expr;
        return DispatchVisit(expr, context);
    }

    /// <summary>
    /// Visit <see cref="Expr"/>.
    /// </summary>
    public TExprResult Visit(Either<AffineConstant, AffineSymbol> expr, TContext context)
    {
        return expr.Match(x => VisitAffineConstant(x, context), x => VisitAffineSymbol(x, context));
    }

    public void Clear()
    {
        VisitRoot = null;
        ExprMemo.Clear();
    }

    /// <summary>
    /// Default visit routine.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    protected internal virtual TExprResult DefaultVisit(AffineExpr expr, TContext context)
    {
        throw new NotImplementedException($"Unhandled visit routine for {expr.GetType()}.");
    }

    /// <summary>
    /// Whether this expression is not visited before.
    /// </summary>
    protected bool HasVisited(AffineExpr expr, [MaybeNullWhen(false)] out TExprResult result)
        => ExprMemo.TryGetValue(expr, out result);

    /// <summary>
    /// Mark expression is visited.
    /// </summary>
    /// <param name="expr">Expression to visit.</param>
    /// <param name="result">Visit result.</param>
    protected TExprResult MarkVisited(AffineExpr expr, TExprResult result)
    {
        ExprMemo[expr] = result;
        return result;
    }

    /// <summary>
    /// Default leaf visit routine.
    /// </summary>
    protected virtual TExprResult DefaultVisitLeaf(AffineExpr expr, TContext context)
    {
        throw new NotImplementedException($"Unhandled visit leaf routine for {expr.GetType()}.");
    }

    protected virtual TExprResult DispatchVisit(AffineExpr expr, TContext context)
    {
        if (HasVisited(expr, out var result))
        {
            return result;
        }

        return MarkVisited(expr, expr.Accept(this, context));
    }
}

/// <summary>
/// Expression visitor.
/// </summary>
/// <typeparam name="TExprResult">Expression visit result type.</typeparam>
public abstract partial class AffineExprVisitor<TExprResult> : AffineExprVisitor<TExprResult, Unit>
{
    /// <summary>
    /// Visit <see cref="Expr"/>.
    /// </summary>
    public TExprResult Visit(AffineExpr expr) => Visit(expr, default);

    /// <summary>
    /// Default visit routine.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Result.</returns>
    protected internal virtual TExprResult DefaultVisit(AffineExpr expr) => base.DefaultVisit(expr, default);

    /// <inheritdoc/>
    protected internal sealed override TExprResult DefaultVisit(AffineExpr expr, Unit context) => DefaultVisit(expr);

    /// <summary>
    /// Default leaf visit routine.
    /// </summary>
    protected virtual TExprResult DefaultVisitLeaf(AffineExpr expr) => base.DefaultVisitLeaf(expr, default);

    protected sealed override TExprResult DefaultVisitLeaf(AffineExpr expr, Unit context) => DefaultVisitLeaf(expr);

    protected virtual TExprResult DispatchVisit(AffineExpr expr) => base.DispatchVisit(expr, default);

    /// <inheritdoc/>
    protected sealed override TExprResult DispatchVisit(AffineExpr expr, Unit context) => DispatchVisit(expr);
}
