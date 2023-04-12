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
public sealed record FusionPattern(Pattern Body, string ModuleKind, VArgsPattern Parameters, string? Name) : Pattern<Fusion>(Name)
{
    /// <inheritdoc/>
    protected override bool MatchLeafCore(Fusion expr)
    {
        return ModuleKind == expr.ModuleKind;
    }
}

public static partial class Utility
{
    /// <summary>
    /// Create the Fusion pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="module_kind">module kind.</param>
    /// <param name="body">body.</param>
    /// <param name="parameters">params.</param>
    /// <returns>FusionPattern .</returns>
    public static FusionPattern IsFusion(string? name, string module_kind, Pattern body, VArgsPattern parameters) => new FusionPattern(body, module_kind, parameters, name);
}
