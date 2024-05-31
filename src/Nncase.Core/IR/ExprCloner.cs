// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Nncase.TIR;

namespace Nncase.IR;

public static class ExprClonerExtensions
{
    public static T Clone<T>(this T expr, bool cloneOtherFunctions = false)
        where T : Expr
    {
        return new ExprCloner<Unit>(cloneOtherFunctions).Clone(expr, default);
    }
}

/// <summary>
/// Expression cloner.
/// </summary>
/// <typeparam name="TContext">Clone context.</typeparam>
public partial class ExprCloner<TContext> : ExprVisitor<Expr, IRType, TContext>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ExprCloner{TContext}"/> class.
    /// </summary>
    /// <param name="cloneOtherFunctions">Clone other functions.</param>
    public ExprCloner(bool cloneOtherFunctions = false)
        : base(cloneOtherFunctions)
    {
    }

    public T Clone<T>(T expr, TContext context)
        where T : Expr
        => (T)Visit(expr, context);

    protected T[] CloneArray<T>(ReadOnlySpan<T> values, TContext context)
        where T : Expr
    {
        var array = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            array[i] = Clone(values[i], context);
        }

        return array;
    }
}

public sealed class ReplacingExprCloner : ExprCloner<Unit>
{
    private readonly IReadOnlyDictionary<Expr, Expr> _replaces;

    public ReplacingExprCloner(IReadOnlyDictionary<Expr, Expr> replaces)
    {
        _replaces = replaces;
    }

    protected override Expr DispatchVisit(Expr expr, Unit context)
    {
        if (_replaces.TryGetValue(expr, out var replacement))
        {
            return replacement;
        }

        return base.DispatchVisit(expr, context);
    }
}
