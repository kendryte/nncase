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
/// <param name="Dimensions">Dimensions condition.</param>
/// <param name="Name">name.</param>
public sealed record ShapePattern(Func<Shape, bool> Condition, VArgsPattern Dimensions, string? Name) : Pattern<Shape>(Name)
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
    /// <returns>ShapePattern.</returns>
    public static ShapePattern IsShape(string? name = null) => IsShape(name, x => true);

    /// <summary>
    /// create the ShapePattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="cond">condition.</param>
    /// <returns>ShapePattern.</returns>
    public static ShapePattern IsShape(string? name, Func<Shape, bool> cond) => new ShapePattern(cond, IsVArgsRepeat(() => IsWildcard()), name);

    /// <summary>
    /// create the ShapePattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="shape">shape.</param>
    /// <returns>ShapePattern.</returns>
    public static ShapePattern IsShape(string? name, Shape shape) => new ShapePattern(x => shape.IsAssignableFrom(shape), IsVArgsRepeat(() => IsWildcard()), name);

    /// <summary>
    /// create the ShapePattern.
    /// </summary>
    /// <param name="shape">shape.</param>
    /// <returns>ShapePattern.</returns>
    public static ShapePattern IsShape(Shape shape) => IsShape(null, shape);

    public static ShapePattern IsShape(Func<Shape, bool> cond) => IsShape(null, cond);

    public static ShapePattern IsRankedShape(string? name = null) => IsShape(name, x => x.IsRanked);

    public static ShapePattern IsFixedShape(string? name = null) => IsShape(name, x => x.IsFixed);
}
