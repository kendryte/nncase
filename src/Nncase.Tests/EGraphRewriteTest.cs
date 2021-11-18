using Xunit;
using System;
using System.Collections.Generic;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Pattern;
using static Nncase.IR.F.Math;
using Rule = Nncase.Transform.Rule;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.Utility;
using System.IO;


public class EGraphRewriteTest : IDisposable
{
    private string dumpPath;
    public EGraphRewriteTest()
    {
        var TestName = System.Reflection.MethodBase.GetCurrentMethod().Name;
        dumpPath = Path.Combine("test_ouput", TestName);
        Directory.CreateDirectory(dumpPath);
    }

    [Fact]
    public void RewriteNoSenceAdd()
    {
        var Name = System.Reflection.MethodBase.GetCurrentMethod().Name;

        Var x = "a";
        var lhs = (x + (100 / 120.0f) - 100);
        var y = lhs + 0;
        var egraph = new EGraph(y);
        EGraphPrinter.DumpEgraphAsDot(egraph, $"{Name}_ADD");

        WildCardPattern wcx = "a";
        var pattern = wcx + IsConst(0);

        // rule  (? + 0) => (?)
        Func<Expr, Expr> nawPass = x => x;

        var EResults = EGraphMatcher.MatchEGraph(egraph, pattern);
        EGraphPrinter.DumpEgraphAsDot(egraph, EResults, $"{Name}_Ematch");
        Assert.Single(EResults);
        var wcxv = EResults[0][wcx];
        Assert.Equal(wcxv, lhs);
        var to_eid = egraph.Add(nawPass(wcxv));

        egraph.Merge(to_eid, egraph.Nodes[((EMatchResult)EResults[0]).Root]);
        EGraphPrinter.DumpEgraphAsDot(egraph, $"{Name}_Merge");
        egraph.ReBuild();
        EGraphPrinter.DumpEgraphAsDot(egraph, $"{Name}_ReBuild");
    }

    [Fact]
    public void TestReassociate()
    {
        Expr expr = ((Const)10 * 11) * 12;
        var eGraph = new EGraph(expr);
        var rule = new Rule.Reassociate();
        EGraphReWrite.ReWriteEGraph(eGraph, rule);
        // Assert.Equal(newExpr, 10 * ((Const)11 * 12));
    }

    public void Dispose()
    {
    }
}
