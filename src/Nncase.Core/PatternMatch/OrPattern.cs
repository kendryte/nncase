// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// The Or Pattern for Match Different branch, NOTE if both branch are matched, choice the Lhs.
/// </summary>
/// <param name="ConditionA">Condition a.</param>
/// <param name="ConditionB">Condition b.</param>
/// <param name="Name">the alt name.</param>
public sealed record OrPattern(Pattern ConditionA, Pattern ConditionB, string? Name)
    : Pattern(Name)
{
    /// <inheritdoc/>
    public override bool MatchLeaf(Expr input) => true;
}

public static partial class Utility
{
    /// <summary>
    /// create or pattern.
    /// </summary>
    public static OrPattern IsAlt(string? name, Pattern condition_a, Pattern condition_b)
        => IsAltImpl(name, condition_a, condition_b);

    /// <summary>
    /// create or pattern without name.
    /// </summary>
    public static OrPattern IsAlt(params Pattern[] patterns)
        => (OrPattern)patterns
            .Aggregate(
                (pattern, pattern1)
                    => IsAltImpl(null, pattern, pattern1));

    private static OrPattern IsAltImpl(string? name, Pattern condition_a, Pattern condition_b)
        => new OrPattern(condition_a, condition_b, name);
}
