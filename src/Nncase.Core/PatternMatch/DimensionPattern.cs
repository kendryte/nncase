﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Dimension"/>.
/// </summary>
/// <param name="Condition">Expression condition.</param>
/// <param name="Name">name.</param>
public sealed record DimensionPattern(Func<Dimension, bool> Condition, string? Name) : Pattern<Dimension>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimensionPattern"/> class.
    /// </summary>
    /// <param name="dimension"><see cref="Dimension"/> expression.</param>
    /// <param name="name">name.</param>
    public DimensionPattern(Dimension dimension, string? name)
        : this(x => x.Equals(dimension), name)
    {
        Value = dimension;
    }

    /// <summary>
    /// Gets value.
    /// </summary>
    public Dimension? Value { get; }

    /// <inheritdoc/>
    protected override bool MatchLeafCore(Dimension expr) => Condition(expr);
}

public static partial class Utility
{
    /// <summary>
    /// create the TensorConstPattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="cond">condition.</param>
    /// <returns>TensorConstPattern.</returns>
    public static DimensionPattern IsDimension(string? name = null, Func<Dimension, bool>? cond = null) => new DimensionPattern(cond ?? (x => true), name);

    public static DimensionPattern IsFixedDimension(string? name = null, long? value = null) => IsDimension(name, x => x.IsFixed && (value == null || x.FixedValue == value));
}
