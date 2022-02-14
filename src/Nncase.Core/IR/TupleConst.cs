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
public sealed record TupleConst(IRArray<Const> Fields) : Const(new TupleType(Fields.Select(x => x.ValueType))),
    IReadOnlyList<Const>, ITuple
{
    /// <summary>
    /// Void type.
    /// </summary>
    public static readonly TupleConst Void = new(ImmutableArray<Const>.Empty);

    /// <inheritdoc/>
    public int Count => Fields.Count;

    IReadOnlyList<Expr> ITuple.Fields => Fields;

    Expr IReadOnlyList<Expr>.this[int index] => this[index];

    /// <summary>
    /// Gets field constant.
    /// </summary>
    /// <param name="index">Index of the field.</param>
    /// <returns>Constant.</returns>
    public Const this[int index] => Fields[index];

    /// <inheritdoc/>
    public IEnumerator<Const> GetEnumerator()
    {
        return Fields.GetEnumerator();
    }

    IEnumerator<Expr> IEnumerable<Expr>.GetEnumerator()
    {
        return GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
