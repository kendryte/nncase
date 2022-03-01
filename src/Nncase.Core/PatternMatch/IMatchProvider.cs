// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Match provider interface.
/// </summary>
public interface IMatchProvider
{
    /// <summary>
    /// Match expression.
    /// </summary>
    /// <param name="expr">Expression to match.</param>
    /// <param name="pattern">Match pattern.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    bool TryMatch(Expr expr, IPattern pattern, [MaybeNullWhen(false)] out IMatchResult result);

    /// <summary>
    /// Match expression as root.
    /// </summary>
    /// <param name="expr">Expression to match.</param>
    /// <param name="pattern">Match pattern.</param>
    /// <param name="result">Match result.</param>
    /// <returns>Match success.</returns>
    bool TryMatchRoot(Expr expr, IPattern pattern, [MaybeNullWhen(false)] out IMatchResult result);
}
