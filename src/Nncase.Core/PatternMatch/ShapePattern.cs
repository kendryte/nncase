// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Shape"/>.
/// </summary>
/// <param name="Condition">Shape condition.</param>
/// <param name="Name">name.</param>
public sealed record ShapePattern(Func<Shape, bool> Condition, string? Name) : Pattern<Shape>(Name)
{
    /// <inheritdoc/>
    protected override bool MatchLeafCore(Shape expr) => Condition(expr);
}

public static partial class Utility
{
    /// <summary>
    /// create the ShapePattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="cond">condition.</param>
    /// <returns>ShapePattern.</returns>
    public static ShapePattern IsShape(string? name = null, Func<Shape, bool>? cond = null) => new ShapePattern(cond ?? (x => true), name);

    /// <summary>
    /// create the ShapePattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="shape">shape.</param>
    /// <returns>ShapePattern.</returns>
    public static ShapePattern IsShape(string? name = null, Shape? shape = null) => new ShapePattern(x => shape?.IsAssignableFrom(shape) ?? true, name);

    public static ShapePattern IsShape(string? name = null) => IsShape(name, shape: null);
}
