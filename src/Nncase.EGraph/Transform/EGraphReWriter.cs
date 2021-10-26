using System;
using System.Text;
using System.IO;
using System.Collections.Generic;
using Nncase.Transform.Rule;
using Nncase.IR;


namespace Nncase.Transform
{

    public class EGraphReWriter
    {
        private bool dumpIr_, isMatchCache_;

        public readonly Dictionary<EGraphRule, List<Expr>> MatchCache = new();

        private string _prefix;

        public EGraphReWriter(bool DumpIr, string Prefix)
        {
            dumpIr_ = DumpIr;
            _prefix = Prefix;
        }

        public bool IsMatchCache(bool value) => isMatchCache_ = value;

        public EGraph Apply(EGraph eGraph, params EGraphRule[] Rules) => Apply(eGraph, new List<EGraphRule>(Rules));

        public EGraph Apply(EGraph eGraph, List<EGraphRule> Rules)
        {
            var eClass = eGraph.EClasses();
            var matches = new List<(EGraphRule, EMatchResult)> { };
            // batch pattern match
            foreach (var rule in Rules)
            {
                var results = EGraphMatcher.EMatch(eClass, rule.GetPatterns());
                foreach (var result in results)
                {
                    matches.Add((rule, result));
                }
                if (dumpIr_)
                    EGraphPrinter.DumpEgraphAsDot(eGraph, results, Path.Combine(_prefix, $"Match_{rule.GetType().Name}"));

            }
            // batch subset 
            Console.WriteLine($"VERSION {eGraph.Version}");
            foreach (var (rule, result) in matches)
            {
                var replaceExpr = rule.GetRePlace(result);
                var neweClass = eGraph.Add(replaceExpr);
                eGraph.Merge(neweClass, eGraph.Nodes[result.Root]);
                if (isMatchCache_)
                {
                    if (MatchCache.ContainsKey(rule))
                        MatchCache[rule].Add(replaceExpr);
                    else
                        MatchCache.Add(rule, new List<Expr> { replaceExpr });
                }
            }
            // batch rebuild
            eGraph.ReBuild();
            if (dumpIr_)
                EGraphPrinter.DumpEgraphAsDot(eGraph, Path.Combine(_prefix, $"Version_{eGraph.Version}"));
            return eGraph;
        }
    }

}