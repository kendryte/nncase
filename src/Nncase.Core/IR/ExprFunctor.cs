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
/// Expression functor.
/// </summary>
/// <typeparam name="TExprResult">Expression visit result type.</typeparam>
/// <typeparam name="TTypeResult">Type visit result type.</typeparam>
/// <typeparam name="TContext">Visit context.</typeparam>
public abstract partial class ExprFunctor<TExprResult, TTypeResult, TContext> : TypeFunctor<TTypeResult, TContext>
{
    /// <summary>
    /// Gets visit root.
    /// </summary>
    protected Expr? VisitRoot { get; private set; }

    /// <summary>
    /// Visit <see cref="Expr"/>.
    /// </summary>
    public TExprResult Visit(Expr expr, TContext context)
    {
        VisitRoot ??= expr;
        return DispatchVisit(expr, context);
    }

    /// <summary>
    /// Clear functor states.
    /// </summary>
    public virtual void Clear()
    {
        VisitRoot = null;
    }

    /// <summary>
    /// Default visit routine.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <param name="context">Context.</param>
    /// <returns>Result.</returns>
    protected internal virtual TExprResult DefaultVisit(Expr expr, TContext context)
    {
        throw new NotImplementedException($"Unhandled visit routine for {expr.GetType()}.");
    }

    protected virtual TExprResult DispatchVisit(Expr expr, TContext context) => expr.Accept(this, context);
}

/// <summary>
/// Expression functor.
/// </summary>
/// <typeparam name="TExprResult">Expression visit result type.</typeparam>
/// <typeparam name="TTypeResult">Type visit result type.</typeparam>
public partial class ExprFunctor<TExprResult, TTypeResult> : ExprFunctor<TExprResult, TTypeResult, Unit>
{
    /// <summary>
    /// Visit <see cref="Expr"/>.
    /// </summary>
    public TExprResult Visit(Expr expr) => Visit(expr, default);

    /// <summary>
    /// Visit type.
    /// </summary>
    /// <param name="type">Type.</param>
    /// <returns>Result.</returns>
    public TTypeResult VisitType(IRType type) => VisitType(type, default);

    /// <summary>
    /// Visit any type.
    /// </summary>
    /// <param name="type">Any type.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult VisitType(AnyType type) => base.VisitType(type, default);

    /// <summary>
    /// Visit None type.
    /// </summary>
    /// <param name="type">None type.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult VisitType(NoneType type) => base.VisitType(type, default);

    /// <summary>
    /// Visit invalid type.
    /// </summary>
    /// <param name="type">Invalid type.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult VisitType(InvalidType type) => base.VisitType(type, default);

    /// <summary>
    /// Visit tensor type.
    /// </summary>
    /// <param name="type">Tensor type.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult VisitType(TensorType type) => base.VisitType(type, default);

    /// <summary>
    /// Visit point type.
    /// </summary>
    /// <param name="type">pointer type.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult VisitType(PointerType type) => base.VisitType(type, default);

    /// <summary>
    /// Visit tuple type.
    /// </summary>
    /// <param name="type">Tuple type.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult VisitType(TupleType type) => base.VisitType(type, default);

    /// <summary>
    /// Visit callable type.
    /// </summary>
    /// <param name="type">Callable type.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult VisitType(CallableType type) => base.VisitType(type, default);

    /// <summary>
    /// Visit callable type.
    /// </summary>
    /// <param name="type">Callable type.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult VisitType(DistributedType type) => base.VisitType(type, default);

    /// <summary>
    /// Default visit routine.
    /// </summary>
    /// <param name="type">Type.</param>
    /// <returns>Result.</returns>
    public virtual TTypeResult DefaultVisitType(IRType type) => base.DefaultVisitType(type, default);

    /// <inheritdoc/>
    public sealed override TTypeResult VisitType(AnyType type, Unit context) => VisitType(type);

    /// <inheritdoc/>
    public sealed override TTypeResult VisitType(NoneType type, Unit context) => VisitType(type);

    /// <inheritdoc/>
    public sealed override TTypeResult VisitType(InvalidType type, Unit context) => VisitType(type);

    /// <inheritdoc/>
    public sealed override TTypeResult VisitType(TensorType type, Unit context) => VisitType(type);

    /// <inheritdoc/>
    public sealed override TTypeResult VisitType(PointerType type, Unit context) => VisitType(type);

    /// <inheritdoc/>
    public sealed override TTypeResult VisitType(TupleType type, Unit context) => VisitType(type);

    /// <inheritdoc/>
    public sealed override TTypeResult VisitType(CallableType type, Unit context) => VisitType(type);

    /// <inheritdoc/>
    public sealed override TTypeResult VisitType(DistributedType type, Unit context) => VisitType(type);

    /// <inheritdoc/>
    public sealed override TTypeResult DefaultVisitType(IRType type, Unit context) => DefaultVisitType(type);

    /// <summary>
    /// Default visit routine.
    /// </summary>
    /// <param name="expr">Expression.</param>
    /// <returns>Result.</returns>
    protected internal virtual TExprResult DefaultVisit(Expr expr) => base.DefaultVisit(expr, default);

    /// <inheritdoc/>
    protected internal sealed override TExprResult DefaultVisit(Expr expr, Unit context) => DefaultVisit(expr);

    protected virtual TExprResult DispatchVisit(Expr expr) => base.DispatchVisit(expr, default);

    /// <inheritdoc/>
    protected sealed override TExprResult DispatchVisit(Expr expr, Unit context) => DispatchVisit(expr);
}
