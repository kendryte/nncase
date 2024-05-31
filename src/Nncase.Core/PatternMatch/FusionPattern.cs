// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Fusion"/>.
/// </summary>
public sealed record FusionPattern(Pattern Body, Func<string, bool> Condition, VArgsPattern Parameters, string? Name) : Pattern<Fusion>(Name)
{
    /// <inheritdoc/>
    protected override bool MatchLeafCore(Fusion expr)
    {
        return Condition(expr.ModuleKind);
    }
}

public static partial class Utility
{
    /// <summary>
    /// Create the Fusion pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="moduleKind">module kind.</param>
    /// <param name="body">body.</param>
    /// <param name="parameters">params.</param>
    /// <returns>FusionPattern .</returns>
    public static FusionPattern IsFusion(string? name, string moduleKind, Pattern body, VArgsPattern parameters) => new FusionPattern(body, m => m == moduleKind, parameters, name);

    public static FusionPattern IsFusion(string? name, Func<string, bool> condition, Pattern body, VArgsPattern parameters) => new FusionPattern(body, condition, parameters, name);
}
