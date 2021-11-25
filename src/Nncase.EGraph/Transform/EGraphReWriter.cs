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
        /// run egraph rewrite
        /// </summary>
        /// <param name="eGraph"></param>
        /// <param name="Rules"></param>
        /// <param name="options"></param>
        /// <returns></returns>
        public static EGraph ReWrite(EGraph eGraph, IEnumerable<PatternRule> Rules, RunPassOptions options)
        {
            var eClass = eGraph.EClasses();
            var matches = new List<(PatternRule, IMatchResult)> { };
            var last_version = eGraph.Version;
            do
            {
                foreach (var rule in Rules)
                {
                    var results = EGraphMatcher.Match(eClass, rule.Patterns);
                    foreach (var result in results)
                    {
                        matches.Add((rule, result));
                    }
                    if (options.DumpLevel > 1)
                        EGraphPrinter.DumpEgraphAsDot(eGraph, results,
                         Path.Combine(options.DumpDir, options.PassName, "Matches", $"{rule.GetType().Name}_V{eGraph.Version}"));
                }
                foreach (var (rule, result) in matches)
                {
                    var replaceExpr = rule.GetRePlace(result);
                    if (replaceExpr is null)
                        continue;
                    if (!TypeInference.InferenceType(replaceExpr))
                        throw new InvalidOperationException("Can't Inference The Replace Expr Type!");
                    eGraph.Add(replaceExpr, out var neweClass);
                    eGraph.Merge(neweClass, eGraph.Nodes[((EMatchResult)result).Root]);
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
            } while (true);
            return eGraph;
        }
    }

}