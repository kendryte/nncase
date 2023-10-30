// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Passes;

public sealed class ExprReplacedEventArgs : EventArgs
{
    public ExprReplacedEventArgs(Expr original, Expr replace)
    {
        Original = original;
        Replace = replace;
    }

    public Expr Original { get; }

    public Expr Replace { get; }
}

/// <summary>
/// Options for running pass.
/// </summary>
public record RunPassContext
{
    /// <summary>
    /// Gets or sets pass index in a <see cref="IPassManager"/>.
    /// </summary>
    public int Index { get; set; }

    /// <summary>
    /// Gets this pass's driver.
    /// </summary>
    public IPass? Driver { get; init; }

    /// <summary>
    /// Gets or sets a value indicating whether control rewrite once or not.
    /// when RewriteOnce is true, the rule will only apply once, then restart rewrite from first rule.
    /// </summary>
    public bool RewriteOnce { get; set; }

    /// <summary>
    /// Gets or sets the match option.
    /// </summary>
    public MatchOptions MatchOptions { get; set; } = new MatchOptions();

    /// <summary>
    /// Gets or sets analysis results.
    /// </summary>
    public IReadOnlyDictionary<Type, IAnalysisResult> AnalysisResults { get; set; } = ImmutableDictionary<Type, IAnalysisResult>.Empty;

    /// <summary>
    /// Gets analysis results.
    /// </summary>
    public T GetAnalysis<T>()
        where T : IAnalysisResult
        => (T)AnalysisResults[typeof(T)];

    /// <summary>
    /// Gets analysis results.
    /// </summary>
    public void GetAnalysis<T>(out T analysis)
        where T : IAnalysisResult
        => analysis = GetAnalysis<T>();
}
