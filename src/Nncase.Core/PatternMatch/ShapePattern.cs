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
/// <param name="Condition">Expression condition.</param>
/// <param name="Name">name.</param>
public sealed record ShapePattern(Func<Shape, bool> Condition, string? Name) : Pattern<Shape>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ShapePattern"/> class.
    /// </summary>
    /// <param name="shape"><see cref="Shape"/> expression.</param>
    /// <param name="name">name.</param>
    public ShapePattern(Shape shape, string? name)
        : this(x => x.Equals(shape), name)
    {
        Value = shape;
    }

    /// <summary>
    /// Gets value.
    /// </summary>
    public Shape? Value { get; }

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
    public static ShapePattern IsShape(string? name = null) => new ShapePattern(x => x is not null, name);

    /// <summary>
    /// create the ShapePattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="cond">condition.</param>
    /// <returns>ShapePattern.</returns>
    public static ShapePattern IsShape(string? name, Func<Shape, bool> cond) => new ShapePattern(cond, name);

    /// <summary>
    /// create the ShapePattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="shape">shape.</param>
    /// <returns>ShapePattern.</returns>
    public static ShapePattern IsShape(string? name, Shape shape) => new ShapePattern(x => shape.IsAssignableFrom(shape), name);

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
