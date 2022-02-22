// using System;
// using System.Collections.Generic;
// using System.IO;
// using System.Linq;
// using Microsoft.Extensions.Hosting;
// using Nncase.Evaluator;
// using Nncase.IR;
// using Nncase.PatternMatch;
// using Nncase.Transform;
// using Xunit;
// using static Nncase.IR.F.Math;
// using static Nncase.IR.F.NN;
// using static Nncase.IR.F.Tensors;
// using static Nncase.PatternMatch.F.Math;
// using static Nncase.PatternMatch.F.NN;
// using static Nncase.PatternMatch.F.Tensors;
// using static Nncase.PatternMatch.Utility;
// using Rule = Nncase.Transform.Rule;


// namespace Nncase.Tests.ReWriteTest
// {
//     public class EGraphRewriteTestFactory : RewriteFixtrue
//     {
//         public EGraphRewriteTestFactory(IHost host) : base(host)
//         {
//             passOptions.SetDir(Path.Combine(passOptions.PassDumpDir, "EGraphRewriteTestFactory"));
//         }

//         private static IEnumerable<object[]> Data =>
//           new List<object[]>
//           {
//              new object[] { new FoldNopClampCase() },
//              new object[] { new FoldNopReshapeCase() },
//              new object[] { new FoldReshapeCase() },
//              new object[] { new TransposeDemoCase() },
//              new object[] { new ClassicDemo() },
//              new object[] { new FoldNopTransposeCase3() },
//              new object[] { new FoldNopTransposeCase2() },
//              new object[] { new FoldNopTransposeCase1() },
//              new object[] { new FoldTransposeCase() },
//              new object[] { new TransposeConstBinaryCase() },
//           };

//         [Theory]
//         [MemberData(nameof(DataOne))]
//         public void RunOne(IRewriteCase Case) => RunCore(Case);

//         protected void RunCore(IRewriteCase Case)
//         {
//             passOptions.SetName($"{Case.Name}");
//             Expr pre = Case.PreExpr;
//             var infered = pre.InferenceType();
//             pre.DumpExprAsIL("pre", passOptions.PassDumpDir);
//             Assert.True(infered);
//             var eGraph = new EGraph();
//             eGraph.Add(pre, out var root);
//             EGraphPrinter.DumpEgraphAsDot(eGraph, Path.Combine(passOptions.PassDumpDir, $"pre"));

//             EGraphReWriter.ReWrite(eGraph, Case.Rules, passOptions);
//             var post = eGraph.Extract(root, passOptions);
//             Assert.True(post.InferenceType());
//             post.DumpExprAsIL("post", passOptions.PassDumpDir);
//             Assert.Equal((pre.Evaluate()), (post.Evaluate()));
//         }

//         [Theory]
//         [MemberData(nameof(DataAll))]
//         public void RunAll(IRewriteCase Case) => RunCore(Case);


//         public static IEnumerable<object[]> DataOne => Data.Take(1);
//         public static IEnumerable<object[]> DataAll => Data.Skip(1);
//     }

//     public class UnitTestEGraphRewrite : RewriteFixtrue
//     {

//         public UnitTestEGraphRewrite(IHost host) : base(host)
//         {
//             passOptions.SetDir(Path.Combine(passOptions.PassDumpDir, "EGraphRewriteTest"));
//         }

//         [Fact]
//         public void RewriteNoSenceAdd()
//         {
//             var Name = System.Reflection.MethodBase.GetCurrentMethod().Name;

//             Var x = "a";
//             var lhs = (x + (100 / 120.0f) - 100);
//             var y = lhs + 0;
//             var egraph = new EGraph(y);
//             EGraphPrinter.DumpEgraphAsDot(egraph, $"{Name}_ADD");

//             WildcardPattern wcx = "a";
//             var pattern = wcx + IsConst(0);

//             // rule  (? + 0) => (?)
//             Func<Expr, Expr> nawPass = x => x;

//             var EResults = EGraphMatcher.Match(egraph, pattern);
//             EGraphPrinter.DumpEgraphAsDot(egraph, EResults, $"{Name}_Ematch");
//             Assert.Single(EResults);
//             var wcxv = EResults[0][wcx];
//             Assert.Equal(wcxv, lhs);
//             egraph.Add(nawPass(wcxv), out var to_eid);

//             egraph.Merge(to_eid, egraph.HashCons[((EMatchResult)EResults[0]).Root]);
//             EGraphPrinter.DumpEgraphAsDot(egraph, $"{Name}_Merge");
//             egraph.ReBuild();
//             EGraphPrinter.DumpEgraphAsDot(egraph, $"{Name}_ReBuild");
//         }

//         [Fact]
//         public void TestReassociate()
//         {
//             Expr pre = ((Const)10 * 11) * 12;
//             var eGraph = new EGraph(pre);
//             var rule = new Rule.ReassociateMul();
//             EGraphReWriter.ReWrite(eGraph, rule, passOptions.SetName("Reassociate"));
//             // Assert.Equal(newExpr, 10 * ((Const)11 * 12));
//         }


//         [Fact]
//         public void TestClassicDemo()
//         {
//             passOptions.SetName("EGraphTest/TestClassicDemo");
//             var g = new EGraph();
//             Var x = "x";
//             g.Add(x * 2, out var e1);
//             g.Add((x * 2) / 2, out var root);
//             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "befroe"));
//             g.Add(x << 1, out var e2);
//             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "added"));
//             g.Merge(e2, e1);
//             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "merge"));
//             g.ReBuild();
//             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "rebuild"));
//         }


//         [Fact]
//         public void TestTransposeBinaryMotion()
//         {
//             passOptions.SetName("TransposeBinaryMotion");
//             Call c0 = (Call)NHWCToNCHW(Tensor.FromScalar(1, new[] { 2, 2, 3, 4 }));
//             Call c1 = (Call)NHWCToNCHW(Tensor.FromScalar(1, new[] { 2, 2, 1, 1 }));
//             Assert.Equal(c0.Parameters[1].GetHashCode(), c1.Parameters[1].GetHashCode());

//             Expr pre = c0 + c1;

//             Assert.True(pre.InferenceType());
//             var eGraph = new EGraph();
//             eGraph.Add(pre, out var root);
//             pre.DumpExprAsIL("pre", passOptions.PassDumpDir);

//             EGraphReWriter.ReWrite(eGraph, new Rule.TransposeBinaryMotion(), passOptions);

//             var post = eGraph.Extract(root, passOptions);
//             Assert.True(post.InferenceType());
//             Assert.Equal((pre.Evaluate()), (post.Evaluate()));
//             post.DumpExprAsIL("post", passOptions.PassDumpDir);
//         }
//     }

// }

