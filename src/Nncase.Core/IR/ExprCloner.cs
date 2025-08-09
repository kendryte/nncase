// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.IR;

public static class ExprClonerExtensions
{
    public static T Clone<T>(this T expr, bool cloneOtherFunctions = false)
        where T : BaseExpr
    {
        return new ExprCloner<Unit>(cloneOtherFunctions).Clone(expr, default);
    }
}

/// <summary>
/// Expression cloner.
/// </summary>
/// <typeparam name="TContext">Clone context.</typeparam>
public partial class ExprCloner<TContext> : ExprVisitor<BaseExpr, IRType, TContext>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ExprCloner{TContext}"/> class.
    /// </summary>
    /// <param name="cloneOtherFunctions">Clone other functions.</param>
    /// <param name="visitAttributes">Visit attributes.</param>
    public ExprCloner(bool cloneOtherFunctions = false, bool visitAttributes = false)
        : base(cloneOtherFunctions, visitAttributes)
    {
    }

    /// <summary>
    /// Gets or sets a value indicating whether clone unmutated, defaults to true.
    /// </summary>
    public bool CloneUnmutated { get; set; } = true;

    public T Clone<T>(T expr, TContext context)
        where T : BaseExpr
        => (T)Visit(expr, context);

    public IVar Clone(IVar expr, TContext context)
        => (IVar)Visit((Expr)expr, context);

    public T CloneType<T>([MaybeNull] T type, TContext context)
        where T : IRType
        => type is null ? null! : (T)VisitType(type, context);

    public override IRType VisitType(AnyType type, TContext context) => type;

    public override IRType VisitType(NoneType type, TContext context) => type;

    public override IRType VisitType(InvalidType type, TContext context) => type;

    public override IRType VisitType(TupleType type, TContext context)
    {
        bool IsOperandsMutated()
        {
            if (IsMutatedTypeArray((ReadOnlySpan<IRType>)type.Fields, context))
            {
                return true;
            }

            return false;
        }

        if (CloneUnmutated || IsOperandsMutated())
        {
            var newTypes = CloneTypeArray((ReadOnlySpan<IRType>)type.Fields, context);
            return type with { Fields = newTypes };
        }

        return type;
    }

    public override IRType VisitType(CallableType type, TContext context)
    {
        bool IsOperandsMutated()
        {
            if (IsMutatedTypeArray((ReadOnlySpan<IRType>)type.Parameters, context)
                || IsMutatedType(type.ReturnType, context))
            {
                return true;
            }

            return false;
        }

        if (CloneUnmutated || IsOperandsMutated())
        {
            var newParamsTypes = CloneTypeArray((ReadOnlySpan<IRType>)type.Parameters, context);
            var newReturnType = CloneType(type.ReturnType, context);
            return type with { Parameters = newParamsTypes, ReturnType = newReturnType };
        }

        return type;
    }

    public override IRType VisitType(PointerType type, TContext context) => type;

    public override IRType VisitType(DistributedType type, TContext context)
    {
        bool IsOperandsMutated()
        {
            if (IsMutatedType(type.TensorType, context))
            {
                return true;
            }

            return false;
        }

        if (CloneUnmutated || IsOperandsMutated())
        {
            var newTensorType = CloneType(type.TensorType, context);
            return type with { TensorType = newTensorType };
        }

        return type;
    }

    public override IRType VisitType(TensorType type, TContext context)
    {
        bool IsOperandsMutated()
        {
            if (IsMutated(type.Shape, context))
            {
                return true;
            }

            return false;
        }

        if (CloneUnmutated || IsOperandsMutated())
        {
            var newShape = Clone(type.Shape, context);
            return type with { Shape = newShape };
        }

        return type;
    }

    public override IRType VisitType(DimensionType type, TContext context) => type;

    public override IRType VisitType(ShapeType type, TContext context) => type;

    public override IRType VisitType(PaddingType type, TContext context) => type;

    public override IRType VisitType(PaddingsType type, TContext context) => type;

    protected T[] CloneArray<T>(ReadOnlySpan<T> values, TContext context)
        where T : BaseExpr
    {
        var array = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            array[i] = Clone(values[i], context);
        }

        return array;
    }

    protected IVar[] CloneArray(ReadOnlySpan<IVar> values, TContext context)
    {
        var array = new IVar[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            array[i] = Clone(values[i], context);
        }

        return array;
    }

    protected T[] CloneTypeArray<T>(ReadOnlySpan<T> values, TContext context)
        where T : IRType
    {
        var array = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            array[i] = CloneType(values[i], context);
        }

        return array;
    }

    protected bool IsMutated<T>(T expr, TContext context)
        where T : BaseExpr
    {
        var newExpr = Visit(expr, context);
        return !ReferenceEquals(expr, newExpr);
    }

    protected bool IsMutated(IVar expr, TContext context) =>
        IsMutated((BaseExpr)expr, context);

    protected bool IsMutatedType<T>([MaybeNull] T type, TContext context)
        where T : IRType
    {
        var newType = type is null ? null : VisitType(type, context);
        return !ReferenceEquals(type, newType);
    }

    protected bool IsMutatedArray<T>(ReadOnlySpan<T> values, TContext context)
        where T : BaseExpr
    {
        return values.AsValueEnumerable().Any(v => IsMutated(v, context));
    }

    protected bool IsMutatedArray(ReadOnlySpan<IVar> values, TContext context)
        => IsMutatedArray(SpanUtility.UnsafeCast<IVar, BaseExpr>(values), context);

    protected bool IsMutatedTypeArray<T>(ReadOnlySpan<T> values, TContext context)
        where T : IRType
    {
        return values.AsValueEnumerable().Any(v => IsMutatedType(v, context));
    }
}

public sealed class ReplacingExprCloner : ExprCloner<Unit>
{
    private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _replaces;

    public ReplacingExprCloner(IReadOnlyDictionary<BaseExpr, BaseExpr> replaces)
    {
        _replaces = replaces;
    }

    protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
    {
        if (_replaces.TryGetValue(expr, out var replacement))
        {
            return replacement;
        }

        return base.DispatchVisit(expr, context);
    }
}
