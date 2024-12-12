// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#define PARALLEL_MATCH

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
using Nncase.Utilities;

namespace Nncase.Passes;

internal class EGraphRewriteProvider : IEGraphRewriteProvider
{
    private readonly ILogger _logger;

    public EGraphRewriteProvider(ILogger<EGraphRewriteProvider> logger)
    {
        _logger = logger;
    }

    public Expr ERewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassContext options, CompileOptions compileOptions)
    {
        if (expr.CheckedType is null)
        {
            CompilerServices.InferenceType(expr);
        }

        var graph = new EGraph(expr);
        ERewrite(graph, rules, options);
        var post = graph.Extract(graph.Root!, compileOptions, null, Array.Empty<EGraphExtractConstrains>());
        return post;
    }

    public IEGraph ERewrite(IEGraph eGraph, IEnumerable<IRewriteRule> rules, RunPassContext context)
    {
        var last_version = eGraph.Version;
        while (true)
        {
            var matches = rules.
#if PARALLEL_MATCH
              AsParallel().
#endif
              Select(rule =>
            {
                EGraphMatcher.TryMatchRoot(eGraph.Nodes, rule.Pattern, out var results);
                return (Rule: rule, Results: results!);
            }).Where(p => p.Results is not null).ToArray();

            if (DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
            {
                using var fs = DumpScope.Current.OpenFile(Path.Combine("Matches", $"V{eGraph.Version}.txt"));
                using var writer = new StreamWriter(fs);
                writer.WriteLine("rule, results");
                foreach (var (rule, results) in matches)
                {
                    writer.WriteLine($"{rule.GetType().Name}, {results.Count}");
                }
            }

            var flattenMatches = (from match in matches
                                  from result in match.Results
                                  select (match.Rule, Result: result)).ToArray();

            var replacedExprs = (from match in flattenMatches.AsParallel()
                                 let oldENode = (ENode)match.Result.Root
                                 let candidates = match.Rule.GetReplaceCandidates(match.Result, context)
                                 where candidates != null
                                 from newExpr in candidates
                                 where newExpr != null
                                 select (oldENode, NewExpr: newExpr.InheritMetaData(oldENode.Expr))).ToArray();

            foreach (var (oldENode, newExpr) in replacedExprs)
            {
                var oldEClass = eGraph.Find(oldENode);
                var typeInferSuccess = CompilerServices.InferenceType(newExpr);
                Trace.Assert(typeInferSuccess);

                var newEClass = eGraph.Add(newExpr);
                if (_logger.IsEnabled(LogLevel.Trace))
                {
                    _logger.LogTrace("Version {Version} : Merge {OldClass} to {NewClass}", eGraph.Version, oldEClass, newEClass);
                }

                eGraph.Union(oldEClass, newEClass);
            }

            if (last_version == eGraph.Version)
            {
                break;
            }
            else
            {
                last_version = eGraph.Version;
            }

            eGraph.Rebuild();
            if (DumpScope.Current.IsEnabled(DumpFlags.Rewrite))
            {
                using var fs = DumpScope.Current.OpenFile(Path.Combine("Rebuild", $"V{eGraph.Version}.dot"));
                EGraphPrinter.DumpEgraphAsDot(eGraph, fs);
            }
        }

        return eGraph;
    }
}
