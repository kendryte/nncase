// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Tuple expression.
/// </summary>
public sealed class Tuple : Expr, ITuple, IEquatable<Tuple?>
{
    public Tuple(ReadOnlySpan<Expr> fields)
        : base(fields.ToArray())
    {
    }

    public Tuple(params Expr[] fields)
        : base(fields.ToArray())
    {
    }

    /// <summary>
    /// Gets void value.
    /// </summary>
    public static TupleConst Void => TupleConst.Void;

    public ReadOnlySpan<Expr> Fields => Operands;

    /// <inheritdoc/>
    public int Count => Fields.Length;

    public Expr this[int index] => Fields[index];

    /// <summary>
    /// cast the value tuple to ir array.
    /// </summary>
    public static implicit operator Tuple(ValueTuple<Expr> tuple) =>
        new Tuple(new Expr[] { tuple.Item1 });

    /// <summary>
    /// cast the value tuple to ir array.
    /// </summary>
    public static implicit operator Tuple((Expr Expr1, Expr Expr2) tuple) =>
        new Tuple(new Expr[] { tuple.Expr1, tuple.Expr2 });

    /// <summary>
    /// cast the value tuple to ir array.
    /// </summary>
    public static implicit operator Tuple((Expr Expr1, Expr Expr2, Expr Expr3) tuple) =>
        new Tuple(new Expr[] { tuple.Expr1, tuple.Expr2, tuple.Expr3 });

    /// <summary>
    /// cast the value tuple to ir array.
    /// </summary>
    public static implicit operator Tuple((Expr Expr1, Expr Expr2, Expr Expr3, Expr Expr4) tuple) =>
        new Tuple(new Expr[] { tuple.Expr1, tuple.Expr2, tuple.Expr3, tuple.Expr4 });

    public static bool operator ==(Tuple? left, Tuple? right) => EqualityComparer<Tuple>.Default.Equals(left, right);

    public static bool operator !=(Tuple? left, Tuple? right) => !(left == right);

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitTuple(this, context);

    public Tuple With(Expr[]? fields = null) => new Tuple(fields ?? Fields);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as Tuple);

    /// <inheritdoc/>
    public bool Equals(Tuple? other) => other is not null && base.Equals(other);
}
