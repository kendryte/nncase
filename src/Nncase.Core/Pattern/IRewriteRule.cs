// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;

namespace Nncase.Pattern;

/// <summary>
/// Rewrite rule.
/// </summary>
public interface IRewriteRule
{
    /// <summary>
    /// Gets pattern.
    /// </summary>
    IPattern Pattern { get; }

    /// <summary>
    /// Get replace expression.
    /// </summary>
    /// <param name="result">Match result.</param>
    /// <returns>Replace expression or null if nothing changed.</returns>
    Expr? GetReplace(IMatchResult result);
}
