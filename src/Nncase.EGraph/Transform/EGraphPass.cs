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
public class EGraphPass : RulesPass
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

    /// <inheritdoc/>
    protected override Callable RunCore(Callable function, RunPassOptions options)
    {
        options.SetPassName(Name);
        var graph = new EGraph();
        var root = graph.Add(function);
        EGraphRewriter.Rewrite(graph, Rules, options);
        var post = graph.Extract(root, options);
        return (Callable)post;
    }
}
