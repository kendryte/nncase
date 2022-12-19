using System.IO;
using Nncase.IR;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.ReWriteTest;


public class UnitTestEGraphRewrite : TestFixture.UnitTestFixtrue
{

    [Fact]
    public void RewriteNoSenceAdd()
    {
        var passOptions = GetPassOptions();
        Var x = "a";
        var lhs = (x + (100 / 120.0f) - 100);
        var y = lhs + 0;

        var pattern = PatternMatch.F.Math.Add(IsWildcard("lhs"), IsConst(0));

        var egraph = new EGraph();
        var root = egraph.Add(y);

        Assert.True(CompilerServices.TryMatchRoot(root.Nodes, pattern, out var EResults));
        EGraphPrinter.DumpEgraphAsDot(egraph, EResults, Path.Combine(passOptions.DumpDir, "Ematch"));
        Assert.Single(EResults);
        var wcxv = (Expr)EResults[0][pattern.Parameters[0]];
        Assert.Equal(wcxv, lhs);
        var to_eid = egraph.Add(wcxv);
        /* 
          lhs + 0 <=> lhs
         */
        egraph.Union(to_eid, root);
        EGraphPrinter.DumpEgraphAsDot(egraph, Path.Combine(passOptions.DumpDir, "Merge"));
        egraph.Rebuild();
        EGraphPrinter.DumpEgraphAsDot(egraph, Path.Combine(passOptions.DumpDir, "ReBuild"));
    }

    [Fact]
    public void TestReassociate()
    {
        var caseOptions = GetPassOptions();
        Expr pre = ((Const)10 * 11) * 12;
        var rule = new Transform.Rules.Neutral.ReassociateMul();
        CompilerServices.ERewrite(pre, new[] { rule }, caseOptions);
        // Assert.Equal(newExpr, 10 * ((Const)11 * 12));
    }


    [Fact]
    public void TestClassicDemo()
    {
        var caseOptions = GetPassOptions();
        var g = new EGraph();
        Var x = "x";
        var e1 = g.Add(x * 2);
        var root = g.Add((x * 2) / 2);
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(caseOptions.DumpDir, "befroe"));
        var e2 = g.Add(x << 1);
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(caseOptions.DumpDir, "added"));
        g.Union(e2, e1);
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(caseOptions.DumpDir, "merge"));
        g.Rebuild();
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(caseOptions.DumpDir, "rebuild"));
    }


    [Fact]
    public void TestTransposeBinaryMotion()
    {
        var caseOptions = GetPassOptions().SetDumpLevel(5);
        Call c0 = (Call)NHWCToNCHW(Tensor.FromScalar(1, new[] { 2, 2, 3, 4 }));
        Call c1 = (Call)NHWCToNCHW(Tensor.FromScalar(1, new[] { 2, 2, 1, 1 }));
        Assert.Equal(c0.Parameters[1].GetHashCode(), c1.Parameters[1].GetHashCode());

        Expr pre = c0 + c1;

        Assert.True(pre.InferenceType());

        var post = CompilerServices.ERewrite(pre, new[] { new Transform.Rules.Neutral.CombineTransposeBinary() }, caseOptions);

        Assert.True(post.InferenceType());
        Assert.Equal((pre.Evaluate()), (post.Evaluate()));
    }
}

