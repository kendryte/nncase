using Xunit;
using System;
using System.Linq;
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


namespace Nncase.Tests.ReWrite
{

    using Evaluator = Evaluator.Evaluator;

    public class EGraphRewriteTestFactory : RewriteTest
    {
        public static IEnumerable<object[]> Data =>
          new List<object[]>
          {
             new object[] { new FoldNopTransposeCase3() },
             new object[] { new FoldNopTransposeCase2() },
             new object[] { new FoldNopTransposeCase1() },
             new object[] { new FoldTransposeCase() },
             new object[] { new TransposeConstBinaryCase() },
          };

        [Theory]
        [MemberData(nameof(DataOne))]
        public void RunOne(IRewriteCase Case) => RunCore(Case);

        public static IEnumerable<object[]> DataOne => Data.Take(1);

        public void RunCore(IRewriteCase Case)
        {
            passOptions.SetName($"EGraphRewriteTest/{Case.Name}");
            Expr pre = Case.PreExpr;
            Assert.True(pre.InferenceType());
            var eGraph = new EGraph();
            eGraph.Add(pre, out var root);
            EGraphPrinter.DumpEgraphAsDot(eGraph, Path.Combine(passOptions.FullDumpDir, $"pre"));
            pre.DumpExprAsIL("pre", passOptions.FullDumpDir);

            EGraphReWriter.ReWrite(eGraph, Case.Rules, passOptions);
            var post = eGraph.Extract(root, passOptions);
            Assert.True(post.InferenceType());
            Assert.Equal(Evaluator.Eval(pre), Evaluator.Eval(post));
            post.DumpExprAsIL("post", passOptions.FullDumpDir);
        }

        [Theory]
        [MemberData(nameof(Data))]
        public void RunAll(IRewriteCase Case) => RunCore(Case);

    }

    public class EGraphRewriteTest : RewriteTest
    {

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
            egraph.Add(nawPass(wcxv), out var to_eid);

            egraph.Merge(to_eid, egraph.HashCons[((EMatchResult)EResults[0]).Root]);
            EGraphPrinter.DumpEgraphAsDot(egraph, $"{Name}_Merge");
            egraph.ReBuild();
            EGraphPrinter.DumpEgraphAsDot(egraph, $"{Name}_ReBuild");
        }

        [Fact]
        public void TestReassociate()
        {
            Expr pre = ((Const)10 * 11) * 12;
            var eGraph = new EGraph(pre);
            var rule = new Rule.Reassociate();
            EGraphReWriter.ReWrite(eGraph, rule, passOptions.SetName("Reassociate"));
            // Assert.Equal(newExpr, 10 * ((Const)11 * 12));
        }


        [Fact]
        public void TestTransposeBinaryMotion()
        {
            passOptions.SetName("TransposeBinaryMotion");
            Call c0 = (Call)NHWCToNCHW(Const.FromShape<int>(new[] { 2, 2, 3, 4 }, 1));
            Call c1 = (Call)NHWCToNCHW(Const.FromShape<int>(new[] { 2, 2, 1, 1 }, 1));
            Assert.Equal(c0.Parameters[1].GetHashCode(), c1.Parameters[1].GetHashCode());

            Expr pre = c0 + c1;

            Assert.True(pre.InferenceType());
            var eGraph = new EGraph();
            eGraph.Add(pre, out var root);
            pre.DumpExprAsIL("pre", passOptions.FullDumpDir);

            EGraphReWriter.ReWrite(eGraph, new Rule.TransposeBinaryMotion(), passOptions);

            var post = eGraph.Extract(root, passOptions);
            Assert.Equal(Evaluator.Eval(pre), Evaluator.Eval(post));
            post.DumpExprAsIL("post", passOptions.FullDumpDir);
        }
    }

}

