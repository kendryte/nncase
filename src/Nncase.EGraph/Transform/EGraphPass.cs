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
/// EGraph pass.
/// </summary>
public class EGraphPass : FunctionPass
{
    private readonly List<IRewriteRule> _rules = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="EGraphPass"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    public EGraphPass(string name)
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
    protected override Callable RunCore(Callable function, RunPassOptions options)
    {
        options.SetPassName(Name);
        var graph = new EGraph();
        graph.Add(function);
        EGraphRewriter.Rewrite(graph, Rules, options);
        return function;
    }
}
