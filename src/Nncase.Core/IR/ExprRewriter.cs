// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    public Expr Rewrite(Expr expr, TContext context)
    {
        using var exprScope = new ExprScope();
        var newExpr = Visit(expr, context);
        DCE(newExpr, exprScope);
        return newExpr;
    }

    /// <summary>
    /// Default rewrite leaf routine.
    /// </summary>
    protected virtual Expr DefaultRewriteLeaf(Expr expr, TContext context) => expr;

    protected void SetMutated() => IsMutated = true;

    protected override void VisitOperands(Expr expr, TContext context)
    {
        var operands = expr.Operands;
        for (int i = 0; i < operands.Length; i++)
        {
            var operand = operands[i];
            var newOperand = Visit(operand, context);
            if (!ReferenceEquals(operand, newOperand))
            {
                expr.ReplaceOperand(i, newOperand);
                SetMutated();
            }
        }
    }

    private void DCE(Expr root, ExprScope exprScope)
    {
        using var exprPin = new ExprPinner(root);
        foreach (var expr in ExprMemo)
        {
            expr.Key.DisposeIfNoUsers();
            expr.Value.DisposeIfNoUsers();
        }

        foreach (var expr in exprScope.Exprs)
        {
            if (expr is not ExprUser)
            {
                expr.DisposeIfNoUsers();
            }
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
    /// Default rewrite leaf routine.
    /// </summary>
    protected virtual Expr DefaultRewriteLeaf(Expr expr) => base.DefaultRewriteLeaf(expr, default);

    /// <inheritdoc/>
    protected sealed override Expr DefaultRewriteLeaf(Expr expr, Unit context) => DefaultRewriteLeaf(expr);
}
