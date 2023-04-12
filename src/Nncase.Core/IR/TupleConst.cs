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
/// Constant of tuple.
/// </summary>
public sealed class TupleConst : Const, ITuple, IEquatable<TupleConst?>
{
    public TupleConst(TupleValue value)
        : base(value.Type)
    {
        Value = value;
    }

    /// <summary>
    /// Gets void value.
    /// </summary>
    public static TupleConst Void => new(TupleValue.Void);

    public TupleValue Value { get; }

    /// <inheritdoc/>
    public int Count => Value.Count;

    /// <summary>
    /// Gets field constant.
    /// </summary>
    /// <param name="index">Index of the field.</param>
    /// <returns>Constant.</returns>
    public IValue this[int index] => Value[index];

    public static bool operator ==(TupleConst? left, TupleConst? right) => EqualityComparer<TupleConst>.Default.Equals(left, right);

    public static bool operator !=(TupleConst? left, TupleConst? right) => !(left == right);

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitTupleConst(this, context);

    public TupleConst With(TupleValue? value = null) => new TupleConst(value ?? Value);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as TupleConst);

    /// <inheritdoc/>
    public bool Equals(TupleConst? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && base.Equals(other) && EqualityComparer<TupleValue>.Default.Equals(Value, other.Value);
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Value);
}
