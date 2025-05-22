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
public abstract partial class ExprRewriter<TContext> : ExprVisitor<BaseExpr, IRType, TContext>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ExprRewriter{TContext}"/> class.
    /// </summary>
    /// <param name="visitOtherFunctions">Vist other functions.</param>
    /// <param name="visitAttributes">Visit attributes.</param>
    public ExprRewriter(bool visitOtherFunctions = false, bool visitAttributes = false)
        : base(visitOtherFunctions, visitAttributes)
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
    public BaseExpr Rewrite(BaseExpr expr, TContext context)
    {
        using var exprScope = new ExprScope();
        var newExpr = Visit(expr, context);
        DCE(newExpr, exprScope);
        return newExpr;
    }

    public override IRType VisitTypeLeaf(AnyType type, TContext context) => type;

    public override IRType VisitTypeLeaf(CallableType type, TContext context) => type;

    public override IRType VisitTypeLeaf(InvalidType type, TContext context) => type;

    public override IRType VisitTypeLeaf(NoneType type, TContext context) => type;

    public override IRType VisitTypeLeaf(TensorType type, TContext context) => type;

    public override IRType VisitTypeLeaf(TupleType type, TContext context) => type;

    public override IRType VisitTypeLeaf(DistributedType type, TContext context) => type;

    public override IRType VisitTypeLeaf(DimensionType type, TContext context) => type;

    public override IRType VisitTypeLeaf(ShapeType type, TContext context) => type;

    public override IRType VisitTypeLeaf(PaddingType type, TContext context) => type;

    public override IRType VisitTypeLeaf(PaddingsType type, TContext context) => type;

    /// <summary>
    /// Default rewrite leaf routine.
    /// </summary>
    protected virtual BaseExpr DefaultRewriteLeaf(BaseExpr expr, TContext context) => expr;

    protected void SetMutated() => IsMutated = true;

    protected override void VisitOperands(BaseExpr expr, TContext context)
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

    protected override void VisitAttributes(BaseExpr expr, TContext context)
    {
        var type = expr.RawCheckedType;
        if (type != null)
        {
            var newType = VisitType(type, context);
            if (!ReferenceEquals(type, newType))
            {
                expr.CheckedType = newType;
                SetMutated();
            }
        }
    }

    private void DCE(BaseExpr root, ExprScope exprScope)
    {
        // using var exprPin = new ExprPinner(root);
        // GC.Collect();
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
    /// <param name="visitAttributes">Visit attributes.</param>
    protected ExprRewriter(bool visitOtherFunctions = false, bool visitAttributes = false)
        : base(visitOtherFunctions, visitAttributes)
    {
    }

    /// <summary>
    /// Rewrite expression.
    /// </summary>
    /// <param name="expr">Expression to rewrite.</param>
    /// <returns>Rewritten expression.</returns>
    public BaseExpr Rewrite(BaseExpr expr) => Rewrite(expr, default);

    /// <summary>
    /// Default rewrite leaf routine.
    /// </summary>
    protected virtual BaseExpr DefaultRewriteLeaf(BaseExpr expr) => base.DefaultRewriteLeaf(expr, default);

    /// <inheritdoc/>
    protected sealed override BaseExpr DefaultRewriteLeaf(BaseExpr expr, Unit context) => DefaultRewriteLeaf(expr);
}
