// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
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
    protected override async Task<BaseFunction> RunCoreAsync(BaseFunction function, RunPassOptions options)
    {
        var graph = new EGraph();
        var root = graph.Add(function);
        EGraphRewriter.Rewrite(graph, Rules, options);
        OnPostRewriteStart(graph, options);
        await OnPostRewrite(graph, options);
        OnPostRewriteEnd(graph, options);
        var post = graph.Extract(root, options);
        return (BaseFunction)post;
    }

    protected virtual Task OnPostRewrite(EGraph graph, RunPassOptions options)
    {
        return Task.CompletedTask;
    }

    /// <summary>
    /// the callback function you can custom process func with run pass options.
    /// </summary>
    /// <param name="callable"> func without run pass.</param>
    /// <param name="options">Options.</param>
    protected virtual void OnPostRewriteStart(EGraph eGraph, RunPassOptions options)
    {
        switch (options.DumpLevel)
        {
            case >= 2:
                EGraphPrinter.DumpEgraphAsDot(
                    eGraph,
                    null,
                    Path.Combine(options.DumpDir, options.PassName, "PostRewriteStart", $"V{eGraph.Version}"));
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
    protected virtual void OnPostRewriteEnd(EGraph eGraph, RunPassOptions options)
    {
        switch (options.DumpLevel)
        {
            case >= 2:
                EGraphPrinter.DumpEgraphAsDot(
                    eGraph,
                    null,
                    Path.Combine(options.DumpDir, options.PassName, "PostRewriteEnd", $"V{eGraph.Version}"));
                break;
            case >= 1:
                break;
            default:
                break;
        }
    }
}
