// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Transform;

public abstract class RulesPass : FunctionPass, IEnumerable<IRewriteRule>
{
    private readonly List<IRewriteRule> _rules = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="RulesPass"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    public RulesPass(string name)
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

    /// <inheritdoc/>
    public IEnumerator<IRewriteRule> GetEnumerator()
    {
        return _rules.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

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
}
