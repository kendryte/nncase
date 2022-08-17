// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform;

/// <summary>
/// Rewrite rule.
/// </summary>
/// <typeparam name="TPattern">Pattern type.</typeparam>
public abstract class RewriteRule<TPattern> : IRewriteRule
    where TPattern : Pattern
{
    /// <summary>
    /// Gets pattern.
    /// </summary>
    public abstract TPattern Pattern { get; }

    IPattern IRewriteRule.Pattern => Pattern;

    /// <inheritdoc/>
    public bool IsMultiBranchSafe { get; init; } = false;

    /// <inheritdoc/>
    bool IRewriteRule.IsMultiBranchSafe() => IsMultiBranchSafe;

    /// <inheritdoc/>
    public abstract Expr? GetReplace(IMatchResult result, RunPassOptions options);
}
