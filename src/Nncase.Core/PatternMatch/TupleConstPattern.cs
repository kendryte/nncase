// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="TupleConst"/>.
/// </summary>
/// <param name="Condition">Expression condition.</param>
/// <param name="Name">name.</param>
public sealed record TupleConstPattern(Func<TupleConst, bool> Condition, string? Name) : Pattern<TupleConst>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TupleConstPattern"/> class.
    /// </summary>
    /// <param name="const"><see cref="Const"/> expression.</param>
    /// <param name="name">name.</param>
    public TupleConstPattern(TupleConst @const, string? name)
        : this(x => x.Equals(@const), name)
    {
        Value = @const;
    }

    /// <summary>
    /// Gets value.
    /// </summary>
    public TupleConst? Value { get; }

    /// <inheritdoc/>
    protected override bool MatchLeafCore(TupleConst expr) => Condition(expr);
}

public static partial class Utility
{
    /// <summary>
    /// create the tupleconst pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <returns>TupleConstPattern.</returns>
    public static TupleConstPattern IsTupleConst(string? name = null) => new(x => true, name);

    /// <summary>
    /// create the tupleconst pattern.
    /// </summary>
    /// <param name="cond">condition.</param>
    /// <param name="name">name.</param>
    /// <returns>TupleConstPattern.</returns>
    public static TupleConstPattern IsTupleConst(Func<TupleConst, bool> cond, string? name = null) => new(cond, name);

    /// <summary>
    /// create the tupleconst pattern.
    /// </summary>
    /// <param name="typePattern">typepattern. </param>
    /// <param name="name">name.</param>
    /// <returns>TupleConstPattern.</returns>
    public static TupleConstPattern IsTupleConst(TypePattern typePattern, string? name = null) => new(x => typePattern.MatchLeaf(x.ValueType), name);
}
