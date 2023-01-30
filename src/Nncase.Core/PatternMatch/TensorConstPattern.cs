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
/// <param name="Name">name.</param>
public sealed record TensorConstPattern(Func<TensorConst, bool> Condition, string? Name) : Pattern<TensorConst>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TensorConstPattern"/> class.
    /// </summary>
    /// <param name="const"><see cref="Const"/> expression.</param>
    /// <param name="name">name.</param>
    public TensorConstPattern(TensorConst @const, string? name)
        : this(x => x.Equals(@const), name)
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
    /// <summary>
    /// create the TensorConstPattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <returns>TensorConstPattern.</returns>
    public static TensorConstPattern IsTensorConst(string? name) => new TensorConstPattern(x => x is not null, name);

    public static TensorConstPattern IsTensorConst() => IsTensorConst(name: null);

    /// <summary>
    /// create the TensorConstPattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="cond">condition.</param>
    /// <returns>TensorConstPattern.</returns>
    public static TensorConstPattern IsTensorConst(string? name, Func<TensorConst, bool> cond) => new TensorConstPattern(cond, name);

    public static TensorConstPattern IsTensorConst(Func<TensorConst, bool> cond) => IsTensorConst(null, cond);

    /// <summary>
    /// create the TensorConstPattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="typePattern">tyeppattern.</param>
    /// <returns>TensorConstPattern.</returns>
    public static TensorConstPattern IsTensorConst(string? name, TypePattern typePattern) => new TensorConstPattern(x => typePattern.MatchLeaf(x.ValueType), name);

    public static TensorConstPattern IsTensorConst(TypePattern typePattern) => IsTensorConst(null, typePattern);
}
