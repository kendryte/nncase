// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Passes;

/// <summary>
/// Rewrite rule.
/// </summary>
/// <typeparam name="TPattern">Pattern type.</typeparam>
public abstract class RewriteRule<TPattern> : IRewriteRule
    where TPattern : Pattern
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RewriteRule{TPattern}"/> class.
    /// </summary>
    public RewriteRule()
    {
        CompileSession = CompileSessionScope.GetCurrentThrowIfNull();
    }

    /// <summary>
    /// Gets pattern.
    /// </summary>
    public abstract TPattern Pattern { get; }

    IPattern IRewriteRule.Pattern => Pattern;

    /// <summary>
    /// Gets compile session.
    /// </summary>
    protected CompileSession CompileSession { get; }

    /// <inheritdoc/>
    public abstract Expr? GetReplace(IMatchResult result, RunPassContext options);
}
