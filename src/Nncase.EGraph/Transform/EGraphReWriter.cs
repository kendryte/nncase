// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Text;
using System.IO;
using System.Collections.Generic;
using Nncase.Transform.Rule;
using Nncase.IR;
using Nncase.Pattern;

namespace Nncase.Transform
{
    public static class EGraphReWriter
    {
        public static EGraph ReWrite(EGraph eGraph, PatternRule Rules, RunPassOptions options) => ReWrite(eGraph, new List<PatternRule>() { Rules }, options);

        /// <summary>
        /// run egraph rewrite.
        /// </summary>
        /// <param name="eGraph"></param>
        /// <param name="Rules"></param>
        /// <param name="options"></param>
        /// <returns></returns>
        public static EGraph ReWrite(EGraph eGraph, IEnumerable<PatternRule> Rules, RunPassOptions options)
        {
            var matches = new List<(PatternRule, IMatchResult)> { };
            var last_version = eGraph.Version;
            int count = 0;
            do
            {
                var eClasses = eGraph.EClasses();
                foreach (var rule in Rules)
                {
                    var results = EGraphMatcher.Match(eClasses, rule.Patterns);
                    foreach (var result in results)
                    {
                        matches.Add((rule, result));
                    }

                    if (options.DumpLevel > 1 && results.Count != 0)
                        EGraphPrinter.DumpEgraphAsDot(eGraph, results,
                         Path.Combine(options.DumpDir, options.PassName, "Matches", $"V{eGraph.Version}_{count++}_{rule.GetType().Name}"));
                }

                foreach (var (rule, result) in matches)
                {
                    var replaceExpr = rule.GetRePlace(result);
                    if (replaceExpr is null)
                        continue;
                    if (!TypeInference.InferenceType(replaceExpr))
                        throw new InvalidOperationException("Can't Inference The Replace Expr Type!");
                    eGraph.Add(replaceExpr, out var neweClass);
                    var oldeClass = eGraph.HashCons[((EMatchResult)result).Root].Find();
                    if (options.DumpLevel == 3)
                        Console.WriteLine($"Version {eGraph.Version} : Merge {{{oldeClass}}} to {{{neweClass}}}");
                    eGraph.Merge(neweClass, oldeClass);
                }

                matches.Clear();
                if (last_version == eGraph.Version)
                    break;
                else
                    last_version = eGraph.Version;
                eGraph.ReBuild();
                if (options.DumpLevel > 1)
                    EGraphPrinter.DumpEgraphAsDot(eGraph,
                     Path.Combine(options.DumpDir, options.PassName, "Rebuild", $"V{eGraph.Version}"));
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
            } while (true);
            return eGraph;
        }
    }
}