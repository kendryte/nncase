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
/// <param name="Dimensions">Dimensions pattern.</param>
/// <param name="Name">name.</param>
public sealed record RankedShapePattern(Func<RankedShape, bool> Condition, VArgsPattern Dimensions, string? Name) : Pattern<RankedShape>(Name)
{
    /// <inheritdoc/>
    protected override bool MatchLeafCore(RankedShape expr) => Condition(expr);
}

public static partial class Utility
{
    /// <summary>
    /// create the RankedShapePattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="cond">condition.</param>
    /// <param name="dimensions">dimensions.</param>
    /// <returns>RankedShapePattern.</returns>
    public static RankedShapePattern IsRankedShape(string? name = null, Func<RankedShape, bool>? cond = null, VArgsPattern? dimensions = null) => new RankedShapePattern(cond ?? (x => true), dimensions ?? IsVArgsRepeat(IsWildcard), name);

    /// <summary>
    /// create the RankedShapePattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="shape">shape.</param>
    /// <returns>RankedShapePattern.</returns>
    public static RankedShapePattern IsRankedShape(string? name = null, RankedShape? shape = null) => new RankedShapePattern(x => shape?.IsAssignableFrom(shape) ?? true, IsVArgsRepeat(IsWildcard), name);

    public static RankedShapePattern IsRankedShape(string? name = null) => IsRankedShape(name, shape: null);

    public static RankedShapePattern IsRankedShape() => IsRankedShape(null, shape: null);

    /// <summary>
    /// create the RankedShapePattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <returns>RankedShapePattern.</returns>
    public static RankedShapePattern IsFixedShape(string? name = null) => IsRankedShape(name, x => x.IsFixed);
}
