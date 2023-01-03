// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.PatternMatch;
using Tensorflow;

namespace Nncase.Transform;

internal class EGraphRewriteProvider : IEGraphRewriteProvider
{
    private readonly ILogger _logger;

    public EGraphRewriteProvider(ILogger logger)
    {
        _logger = logger;
    }

    public Expr ERewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options)
    {
        if (expr.CheckedType is null)
        {
            CompilerServices.InferenceType(expr);
        }

        var graph = new EGraph();
        var root = graph.Add(expr);
        ERewrite(graph, rules, options);
        var post = graph.Extract(root, null, options);
        return post;
    }

    public IEGraph ERewrite(IEGraph eGraph, IEnumerable<IRewriteRule> rules, RunPassContext context)
    {
        var matches = new List<(IRewriteRule, IReadOnlyList<IMatchResult>)> { };
        var last_version = eGraph.Version;
        int count = 0;

        while (true)
        {
            foreach (var rule in rules)
            {
                if (EGraphMatcher.TryMatchRoot(eGraph.Nodes, rule.Pattern, out var results))
                {
                    matches.Add((rule, results));

                    if (context.Dumpper.IsEnabled(DumpFlags.Rewrite) && results.Count != 0)
                    {
                        using var fs = context.Dumpper.OpenWrite(Path.Combine("Matches", $"V{eGraph.Version}_{count++}_{rule.GetType().Name}.dot"));
                        EGraphPrinter.DumpEgraphAsDot(eGraph, results, fs);
                    }
                }
            }

            foreach (var (rule, results) in matches)
            {
                var replacedExprs = (from result in results
                                     let expr = rule.GetReplace(result, context)
                                     where expr != null
                                     select (eGraph.Find((ENode)result.Root), expr)).ToList();

                foreach (var (oldEClass, newExpr) in replacedExprs)
                {
                    var typeInferSuccess = CompilerServices.InferenceType(newExpr);
                    Trace.Assert(typeInferSuccess);

                    var newEClass = eGraph.Add(newExpr);
                    if (_logger.IsEnabled(LogLevel.Trace))
                    {
                        _logger.LogTrace("Version {Version} : Merge {OldClass} to {NewClass}", eGraph.Version, oldEClass, newEClass);
                    }

                    eGraph.Union(newEClass, oldEClass);
                }
            }

            matches.Clear();
            if (last_version == eGraph.Version)
            {
                break;
            }
            else
            {
                last_version = eGraph.Version;
            }

            eGraph.Rebuild();
            if (context.Dumpper.IsEnabled(DumpFlags.Rewrite))
            {
                using var fs = context.Dumpper.OpenWrite(Path.Combine("Rebuild", $"V{eGraph.Version}.dot"));
                EGraphPrinter.DumpEgraphAsDot(eGraph, fs);
            }
        }

        return eGraph;
    }
}
