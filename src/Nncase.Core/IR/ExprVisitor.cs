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
/// Expression visitor.
/// </summary>
/// <typeparam name="TExprResult">Expression visit result type.</typeparam>
/// <typeparam name="TTypeResult">Type visit result type.</typeparam>
/// <typeparam name="TContext">Visit context.</typeparam>
public abstract partial class ExprVisitor<TExprResult, TTypeResult, TContext> : ExprFunctor<TExprResult, TTypeResult, TContext>
{
    private readonly bool _visitOtherFunctions;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExprVisitor{TExprResult, TTypeResult, TContext}"/> class.
    /// </summary>
    /// <param name="visitOtherFunctions">Vist other functions.</param>
    public ExprVisitor(bool visitOtherFunctions = false)
    {
        _visitOtherFunctions = visitOtherFunctions;
    }

    /// <summary>
    /// Gets expression memo.
    /// </summary>
    public Dictionary<Expr, TExprResult> ExprMemo { get; } = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    /// Gets type memo.
    /// </summary>
    public Dictionary<IRType, TTypeResult> TypeMemo { get; } = new(ReferenceEqualityComparer.Instance);

    /// <inheritdoc/>
    public override TTypeResult VisitType(AnyType type, TContext context)
    {
        if (HasVisited(type, out var result))
        {
            return result;
        }

        return MarkVisited(type, VisitTypeLeaf(type, context));
    }

    /// <inheritdoc/>
    public override TTypeResult VisitType(CallableType type, TContext context)
    {
        if (HasVisited(type, out var result))
        {
            return result;
        }

        foreach (var param in type.Parameters)
        {
            VisitType(param, context);
        }

        VisitType(type.ReturnType, context);
        return MarkVisited(type, VisitTypeLeaf(type, context));
    }

    /// <inheritdoc/>
    public override TTypeResult VisitType(InvalidType type, TContext context)
    {
        if (HasVisited(type, out var result))
        {
            return result;
        }

        return MarkVisited(type, VisitTypeLeaf(type, context));
    }

    /// <inheritdoc/>
    public override TTypeResult VisitType(TensorType type, TContext context)
    {
        if (HasVisited(type, out var result))
        {
            return result;
        }

        return MarkVisited(type, VisitTypeLeaf(type, context));
    }

    /// <inheritdoc/>
    public override TTypeResult VisitType(TupleType type, TContext context)
    {
        if (HasVisited(type, out var result))
        {
            return result;
        }

        foreach (var field in type.Fields)
        {
            VisitType(field, context);
        }

        return MarkVisited(type, VisitTypeLeaf(type, context));
    }

    /// <summary>
    /// Visit any type leaf.
    /// </summary>
    public virtual TTypeResult VisitTypeLeaf(AnyType type, TContext context) => DefaultVisitTypeLeaf(type, context);

    /// <summary>
    /// Visit invalid type leaf.
    /// </summary>
    public virtual TTypeResult VisitTypeLeaf(InvalidType type, TContext context) => DefaultVisitTypeLeaf(type, context);

    /// <summary>
    /// Visit tensor type leaf.
    /// </summary>
    public virtual TTypeResult VisitTypeLeaf(TensorType type, TContext context) => DefaultVisitTypeLeaf(type, context);

    /// <summary>
    /// Visit tuple type leaf.
    /// </summary>
    public virtual TTypeResult VisitTypeLeaf(TupleType type, TContext context) => DefaultVisitTypeLeaf(type, context);

    /// <summary>
    /// Visit tuple type leaf.
    /// </summary>
    public virtual TTypeResult VisitTypeLeaf(CallableType type, TContext context) => DefaultVisitTypeLeaf(type, context);

    /// <summary>
    /// Default visit leaf routine.
    /// </summary>
    public virtual TTypeResult DefaultVisitTypeLeaf(IRType type, TContext context)
    {
        throw new NotImplementedException($"Unhandled visit leaf routine for {type.GetType()}.");
    }

    /// <inheritdoc/>
    public override void Clear()
    {
        ExprMemo.Clear();
        TypeMemo.Clear();
        base.Clear();
    }

    /// <summary>
    /// Whether this expression is not visited before.
    /// </summary>
    protected bool HasVisited(Expr expr, [MaybeNullWhen(false)] out TExprResult result)
        => ExprMemo.TryGetValue(expr, out result);

    /// <summary>
    /// Whether this type is not visited before.
    /// </summary>
    protected bool HasVisited(IRType type, [MaybeNullWhen(false)] out TTypeResult result)
        => TypeMemo.TryGetValue(type, out result);

    /// <summary>
    /// Mark expression is visited.
    /// </summary>
    /// <param name="expr">Expression to visit.</param>
    /// <param name="result">Visit result.</param>
    protected TExprResult MarkVisited(Expr expr, TExprResult result)
    {
        ExprMemo[expr] = result;
        return result;
    }

    /// <summary>
    /// Mark type is visited.
    /// </summary>
    /// <param name="type">Type to visit.</param>
    /// <param name="result">Visit result.</param>
    protected TTypeResult MarkVisited(IRType type, TTypeResult result)
    {
        TypeMemo[type] = result;
        return result;
    }

    protected bool CanVisitFunctionBody(BaseFunction baseFunction)
    {
        if (_visitOtherFunctions)
        {
            return true;
        }

        return ReferenceEquals(baseFunction, VisitRoot);
    }

    /// <summary>
    /// Default leaf visit routine.
    /// </summary>
    protected virtual TExprResult DefaultVisitLeaf(Expr expr, TContext context)
    {
        throw new NotImplementedException($"Unhandled visit leaf routine for {expr.GetType()}.");
    }

    /// <inheritdoc/>
    protected override TExprResult DispatchVisit(Expr expr, TContext context)
    {
        if (HasVisited(expr, out var result))
        {
            return result;
        }

        return MarkVisited(expr, base.DispatchVisit(expr, context));
    }

    protected virtual void VisitOperands(Expr expr, TContext context)
    {
        foreach (var operand in expr.Operands)
        {
            Visit(operand, context);
        }
    }
}

/// <summary>
/// Expression visitor.
/// </summary>
/// <typeparam name="TExprResult">Expression visit result type.</typeparam>
/// <typeparam name="TTypeResult">Type visit result type.</typeparam>
public abstract partial class ExprVisitor<TExprResult, TTypeResult> : ExprVisitor<TExprResult, TTypeResult, Unit>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ExprVisitor{TExprResult, TTypeResult}"/> class.
    /// </summary>
    /// <param name="visitOtherFunctions">Vist other functions.</param>
    public ExprVisitor(bool visitOtherFunctions = false)
        : base(visitOtherFunctions)
    {
    }

    /// <summary>
    /// Visit <see cref="Expr"/>.
    /// </summary>
    public TExprResult Visit(Expr expr) => Visit(expr, default);

    /// <summary>
    /// Default visit routine.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Result.</returns>
    protected internal virtual TExprResult DefaultVisit(Expr expr) => base.DefaultVisit(expr, default);

    /// <inheritdoc/>
    protected internal sealed override TExprResult DefaultVisit(Expr expr, Unit context) => DefaultVisit(expr);

    /// <summary>
    /// Default leaf visit routine.
    /// </summary>
    protected virtual TExprResult DefaultVisitLeaf(Expr expr) => base.DefaultVisitLeaf(expr, default);

    protected sealed override TExprResult DefaultVisitLeaf(Expr expr, Unit context) => DefaultVisitLeaf(expr);

    protected virtual TExprResult DispatchVisit(Expr expr) => base.DispatchVisit(expr, default);

    /// <inheritdoc/>
    protected sealed override TExprResult DispatchVisit(Expr expr, Unit context) => DispatchVisit(expr);
}
