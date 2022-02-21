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
public sealed record TupleConstPattern(Func<TupleConst, bool> Condition) : Pattern<TupleConst>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TupleConstPattern"/> class.
    /// </summary>
    /// <param name="const"><see cref="Const"/> expression.</param>
    public TupleConstPattern(TupleConst @const)
        : this(x => x.Equals(@const))
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
    public static TupleConstPattern IsTupleConst() => new(x => true);

    public static TupleConstPattern IsTupleConst(Func<TupleConst, bool> Cond) => new(Cond);

    public static TupleConstPattern IsTupleConst(TypePattern typePattern) => new(x => typePattern.MatchLeaf(x.ValueType));
}
