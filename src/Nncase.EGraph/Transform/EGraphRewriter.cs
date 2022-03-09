// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform;

internal static class EGraphRewriter
{
    public static EGraph Rewrite(EGraph eGraph, IRewriteRule rules, RunPassOptions options) => Rewrite(eGraph, new List<IRewriteRule>() { rules }, options);

    /// <summary>
    /// Run egraph rewrite.
    /// </summary>
    public static EGraph Rewrite(EGraph eGraph, IEnumerable<IRewriteRule> rules, RunPassOptions options)
    {
        var matches = new List<(IRewriteRule, IReadOnlyList<IMatchResult>)> { };
        var last_version = eGraph.Version;
        int count = 0;

        while (true)
        {
            var enodes = eGraph.HashCons.Keys.ToList();
            var eClasses = eGraph.EClasses();
            foreach (var rule in rules)
            {
                if (EGraphMatcher.TryMatchRoot(enodes, rule.Pattern, out var results))
                {
                    matches.Add((rule, results));

                    if (options.DumpLevel > 1 && results.Count != 0)
                    {
                        EGraphPrinter.DumpEgraphAsDot(
                            eGraph,
                            results,
                            Path.Combine(options.DumpDir, options.PassName, "Matches", $"V{eGraph.Version}_{count++}_{rule.GetType().Name}"));
                    }
                }
            }

            foreach (var (rule, results) in matches)
            {
                var replacedExprs = (from result in results
                                     let expr = rule.GetReplace(result)
                                     where expr != null
                                     select (eGraph.HashCons[(ENode)result.Root], expr)).ToList();

                foreach (var (oldEClass, newExpr) in replacedExprs)
                {
                    if (!CompilerServices.InferenceType(newExpr))
                    {
                        throw new InvalidOperationException("Can't Inference The Replace Expr Type!");
                    }

                    var newEClass = eGraph.Add(newExpr);
                    if (options.DumpLevel == 3)
                    {
                        Console.WriteLine($"Version {eGraph.Version} : Merge {{{oldEClass}}} to {{{newEClass}}}");
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
            if (options.DumpLevel > 1)
            {
                EGraphPrinter.DumpEgraphAsDot(eGraph,
                 Path.Combine(options.DumpDir, options.PassName, "Rebuild", $"V{eGraph.Version}"));
            }

            if (options.DumpLevel == 3)
            {
                foreach (var (_, eclass) in eGraph.HashCons)
                {
                    if (eclass.Parent is not null)
                    {
                        // throw new InvalidProgramException("EGraph Rebuild Logic Error!");
                    }
                }
            }
        }

        return eGraph;
    }
}
