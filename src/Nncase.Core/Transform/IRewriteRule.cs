// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform;

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
    Expr? GetReplace(IMatchResult result, RunPassOptions options);

    /// <summary>
    /// check this pattern can be modify in multi branch.
    /// </summary>
    /// <returns></returns>
    bool IsMultiBranchSafe()
    {
        return false;
    }
}

/// <summary>
/// the attrbuite mark the need auto generator the GetReplace overwrite function.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
public sealed class RuleGeneratorAttribute : Attribute
{
}
