// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.Pattern;

/// <summary>
/// The Or Pattern for Match Different branch, NOTE if both branch are matched, choice the Lhs.
/// </summary>
/// <typeparam name="TInput">Input type.</typeparam>
/// <param name="ConditionA">Condition a.</param>
/// <param name="ConditionB">Condition b.</param>
public sealed record OrPattern<TInput>(IPattern<TInput> ConditionA, IPattern<TInput> ConditionB)
    : Pattern, IPattern<TInput>
{
    /// <inheritdoc/>
    public bool Match(TInput input)
    {
        return ConditionA.Match(input) || ConditionB.Match(input);
    }

    /// <inheritdoc/>
    public sealed override bool Match(object input) => input is TInput expr && Match(expr);
}
