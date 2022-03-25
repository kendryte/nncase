// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform;

/// <summary>
/// Dataflow pass.
/// </summary>
public class DataflowPass : FunctionPass
{
    private readonly List<IRewriteRule> _rules = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="DataflowPass"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    public DataflowPass(string name)
        : base(name)
    {
    }

    /// <summary>
    /// Gets rules.
    /// </summary>
    public IReadOnlyList<IRewriteRule> Rules => _rules;

    /// <summary>
    /// add the pattern rule.
    /// </summary>
    /// <param name="rule">Rule.</param>
    public void Add(IRewriteRule rule) => _rules.Add(rule);

    /// <summary>
    /// add the pattern rules.
    /// </summary>
    /// <param name="rules">Rules.</param>
    public void Add(params IRewriteRule[] rules) => _rules.AddRange(rules);

    /// <summary>
    /// <see cref="Add(IRewriteRule[])"/>.
    /// </summary>
    /// <param name="rules">Rules.</param>
    public void Add(IEnumerable<IRewriteRule> rules) => _rules.AddRange(rules);

    /// <summary>
    /// the callback function you can custom process func with run pass options.
    /// </summary>
    /// <param name="callable"> func without run pass.</param>
    /// <param name="options">Options.</param>
    protected override void OnPassStart(Callable callable, RunPassOptions options)
    {
        switch (options.DumpLevel)
        {
            case >= 2:
                CompilerServices.DumpIR(callable, "Start", Path.Combine(options.PassDumpDir, Name));
                break;
            case >= 1:
                break;
            default:
                break;
        }
    }

    /// <summary>
    /// the callback function you can custom process func with run pass options.
    /// </summary>
    /// <param name="callable"> func with rewrited. </param>
    /// <param name="options">Options.</param>
    protected override void OnPassEnd(Callable callable, RunPassOptions options)
    {
        switch (options.DumpLevel)
        {
            case >= 2:
                CompilerServices.DumpIR(callable, "End", Path.Combine(options.PassDumpDir, Name));
                break;
            case >= 1:
                break;
            default:
                break;
        }
    }

    /// <inheritdoc/>
    protected override Callable RunCore(Callable pre, RunPassOptions options)
    {
        OnPassStart(pre, options);
        Function post = (Function)CompilerServices.Rewrite(pre, Rules, options);
        OnPassEnd(post, options);
        return post;
    }
}
