// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="TensorConst"/>.
/// </summary>
/// <param name="Condition">Expression condition.</param>
public sealed record TensorConstPattern(Func<TensorConst, bool> Condition) : Pattern<TensorConst>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TensorConstPattern"/> class.
    /// </summary>
    /// <param name="const"><see cref="Const"/> expression.</param>
    public TensorConstPattern(TensorConst @const)
        : this(x => x.Equals(@const))
    {
        Value = @const;
    }

    /// <summary>
    /// Gets value.
    /// </summary>
    public TensorConst? Value { get; }

    /// <inheritdoc/>
    protected override bool MatchLeafCore(TensorConst expr) => Condition(expr);
}

public static partial class Utility
{
    public static TensorConstPattern IsTensorConst() => new TensorConstPattern(x => x is TensorConst);

    public static TensorConstPattern IsTensorConst(Func<TensorConst, bool> Cond) => new TensorConstPattern(Cond);

    public static TensorConstPattern IsTensorConst(TypePattern typePattern) => new TensorConstPattern(x => typePattern.MatchLeaf(x.ValueType));
}
