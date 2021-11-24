using Xunit;
using System;
using System.Collections.Generic;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Pattern;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;
using Rule = Nncase.Transform.Rule;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.NN;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.Utility;
using System.IO;
using System.Runtime.CompilerServices;



public class EGraphRewriteTest : IDisposable
{
    private RunPassOptions _options;

    public EGraphRewriteTest()
    {
        var TestName = this.GetType().Name;
        string dumpDir = Path.Combine(GetThisFilePath(), "..", "..", "..", "tests_output");
        dumpDir = Path.GetFullPath(dumpDir);
        Directory.CreateDirectory(dumpDir);
        _options = new RunPassOptions(null, 3, dumpDir);
    }

    private static string GetThisFilePath([CallerFilePath] string path = null)
    {
        return path;
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

        var EResults = EGraphMatcher.Match(egraph, pattern);
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
        EGraphReWriter.ReWrite(eGraph, rule, _options.SetName("Reassociate"));
        // Assert.Equal(newExpr, 10 * ((Const)11 * 12));
    }


    [Fact]
    public void TestTransposeBinaryMotion()
    {
        Call c0 = (Call)NHWCToNCHW(Const.FromShape<int>(new[] { 2, 2, 3, 4 }, 1));
        Call c1 = (Call)NHWCToNCHW(Const.FromShape<int>(new[] { 2, 2, 1, 1 }, 1));
        Assert.Equal(c0.Parameters[1].GetHashCode(), c1.Parameters[1].GetHashCode());

        Expr expr = c0 + c1;
        var eGraph = new EGraph(expr);
        EGraphPrinter.DumpEgraphAsDot(eGraph, Path.Combine(_options.DumpDir, "ir_import"));
        EGraphReWriter.ReWrite(eGraph, new Rule.TransposeBinaryMotion(), _options.SetName("TransposeBinaryMotion"));
        
    }

    [Fact]
    public void TestTransposeBinaryAll()
    {

    }

    public void Dispose()
    {
    }
}
