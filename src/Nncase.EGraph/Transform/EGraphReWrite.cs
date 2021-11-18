using System;
using System.Text;
using System.IO;
using System.Collections.Generic;
using Nncase.Transform.Rule;
using Nncase.IR;
using Nncase.Pattern;


namespace Nncase.Transform
{

    public static class EGraphReWrite
    {
        public static EGraph ReWriteEGraph(EGraph eGraph, params PatternRule[] Rules) => ReWriteEGraph(eGraph, new List<PatternRule>(Rules));

        public static EGraph ReWriteEGraph(EGraph eGraph, List<PatternRule> Rules, bool DumpMatches = false, string prefix = "")
        {
            var eClass = eGraph.EClasses();
            var matches = new List<(PatternRule, IMatchResult)> { };
            // batch pattern match
            foreach (var rule in Rules)
            {
                var results = EGraphMatcher.MatchEGraph(eClass, rule.Patterns);
                foreach (var result in results)
                {
                    matches.Add((rule, result));
                }
                if (DumpMatches)
                    EGraphPrinter.DumpEgraphAsDot(eGraph, results, Path.Combine(prefix, $"{rule.GetType().Name}_Matches"));
            }
            // batch subset 
            Console.WriteLine($"VERSION {eGraph.Version}");
            foreach (var (rule, result) in matches)
            {
                var replaceExpr = rule.GetRePlace(result);
                if (replaceExpr is null)
                    continue;
                var neweClass = eGraph.Add(replaceExpr);
                eGraph.Merge(neweClass, eGraph.Nodes[((EMatchResult)result).Root]);
            }
            // batch rebuild
            eGraph.ReBuild();
            // if (dumpIr_)
            //     EGraphPrinter.DumpEgraphAsDot(eGraph, Path.Combine(_prefix, $"Version_{eGraph.Version}"));
            return eGraph;
        }
    }

}