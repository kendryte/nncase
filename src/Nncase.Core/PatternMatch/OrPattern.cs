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
/// <typeparam name="TInput">Input type.</typeparam>
/// <param name="ConditionA">Condition a.</param>
/// <param name="ConditionB">Condition b.</param>
public sealed record OrPattern(Pattern ConditionA, Pattern ConditionB, string? Name)
    : Pattern(Name)
{
    /// <inheritdoc/>
    public override bool MatchLeaf(object input) => true;
}


public static partial class Utility
{
    public static OrPattern IsAlt(string? name, Pattern condition_a, Pattern condition_b)
       => new OrPattern(condition_a, condition_b, name);

    public static OrPattern IsAlt(Pattern condition_a, Pattern condition_b)
       => IsAlt(null, condition_a, condition_b);
}