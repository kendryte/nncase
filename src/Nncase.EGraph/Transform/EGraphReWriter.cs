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

        public static EGraph ReWrite(EGraph eGraph, params PatternRule[] Rules) => ReWrite(eGraph, new List<PatternRule>(Rules), new RunPassOptions(null, 2, "EGraphPass"));

        /// <summary>
        /// run egraph rewrite
        /// </summary>
        /// <param name="eGraph"></param>
        /// <param name="Rules"></param>
        /// <param name="options"></param>
        /// <returns></returns>
        public static EGraph ReWrite(EGraph eGraph, List<PatternRule> Rules, RunPassOptions options)
        {
            var eClass = eGraph.EClasses();
            var matches = new List<(PatternRule, IMatchResult)> { };
            foreach (var rule in Rules)
            {
                var results = EGraphMatcher.Match(eClass, rule.Patterns);
                foreach (var result in results)
                {
                    matches.Add((rule, result));
                }
                if (options.DumpLevel > 1)
                    EGraphPrinter.DumpEgraphAsDot(eGraph, results,
                     Path.Combine(options.DumpDir, options.PassName, "Matches", $"{rule.GetType().Name}_{eGraph.Version}"));
            }
            foreach (var (rule, result) in matches)
            {
                var replaceExpr = rule.GetRePlace(result);
                if (replaceExpr is null)
                    continue;
                var neweClass = eGraph.Add(replaceExpr);
                eGraph.Merge(neweClass, eGraph.Nodes[((EMatchResult)result).Root]);
            }
            eGraph.ReBuild();
            if (options.DumpLevel > 1)
                EGraphPrinter.DumpEgraphAsDot(eGraph,
                 Path.Combine(options.DumpDir, options.PassName, "Rebuild", $"{eGraph.Version}"));
            return eGraph;
        }
    }

}