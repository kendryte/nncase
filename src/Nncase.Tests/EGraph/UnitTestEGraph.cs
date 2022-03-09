using System;
using System.IO;
using System.Linq;
using Xunit;
using Nncase.Transform;
using Nncase.IR;
using static Nncase.IR.F.Math;
using Nncase.IR.Math;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Nncase.Tests.EGraphTest
{
    public class UnitTestEGraph
    {
        public RunPassOptions passOptions;

        private static string GetThisFilePath([CallerFilePath] string path = null)
        {
            return path;
        }

        public UnitTestEGraph()
        {
            var TestName = this.GetType().Name;
            string dumpDir = Testing.GetDumpDirPath(typeof(UnitTestEGraph));
            dumpDir = Path.GetFullPath(dumpDir);
            Directory.CreateDirectory(dumpDir);
            passOptions = new RunPassOptions(null, 3, dumpDir);
        }

        [Fact]
        public void TestEqualEClass()
        {
            var a = 1 + Exp(2);
            var b = 1 + Exp(2);

            var graph = new EGraph();
            Assert.StrictEqual(graph.Add(a), graph.Add(b));
        }

        [Fact]
        public void TestENodeVarHash()
        {
            var node1 = new ENode((Var)"x", new EClass[] { });
            var node2 = new ENode((Var)"x", new EClass[] { });
            var node3 = new ENode((Var)"y", new EClass[] { });
            Assert.Equal(node1, node2);
            Assert.Equal(node1.GetHashCode(), node2.GetHashCode());
            Assert.NotEqual(node1, node3);
            Assert.NotEqual(node1.GetHashCode(), node3.GetHashCode());
        }

        [Fact]
        public void TestENodeConstHash()
        {
            var node1 = new ENode(1, new EClass[] { });
            var node2 = new ENode(1, new EClass[] { });
            var node3 = new ENode(11, new EClass[] { });
            Assert.Equal(node1, node2);
            Assert.Equal(node1.GetHashCode(), node2.GetHashCode());
            Assert.NotEqual(node1, node3);
            Assert.NotEqual(node1.GetHashCode(), node3.GetHashCode());
        }

        [Fact]
        public void TestENodeOpHash()
        {
            var node1 = new ENode(new Binary(BinaryOp.Add), new EClass[] { });
            var node2 = new ENode(new Binary(BinaryOp.Add), new EClass[] { });
            var node3 = new ENode(new Unary(UnaryOp.Abs), new EClass[] { });
            Assert.Equal(node1, node2);
            Assert.Equal(node1.GetHashCode(), node2.GetHashCode());
            Assert.NotEqual(node1, node3);
            Assert.NotEqual(node1.GetHashCode(), node3.GetHashCode());
        }

        [Fact]
        public void TestENodeCallHash()
        {
            // when the expr have same eclass args, but their expr is not
            // the Enode must be equal
            var eclass = new EClass(1);
            var call1 = Binary(BinaryOp.Add, 4, 4);
            var call2 = Binary(BinaryOp.Add, ((Const)1 + 3), ((Const)2 + 2));
            var node1 = new ENode(call1, new[] { eclass, eclass });
            var node2 = new ENode(call2, new[] { eclass, eclass });
            Assert.Equal(node1.GetHashCode(), node2.GetHashCode());
            Assert.Equal(node1, node2);
        }

        [Fact]
        public void TestENodeCallHashEGraph()
        {
            var egraph = new EGraph();
            var call1 = Binary(BinaryOp.Add, 4, 4);
            var call2 = Binary(BinaryOp.Add, (Const)1 + 3, (Const)2 + 2);
            var e1 = egraph.Add(4);
            var e2 = egraph.Add((Const)1 + 3);
            var e3 = egraph.Add((Const)2 + 2);
            var e4 = egraph.Add(call1);

            egraph.Union(e1, e2);
            egraph.Union(e1, e3);
            egraph.Rebuild();

            var e5 = egraph.Add(call2);
            Assert.StrictEqual(e5.Find(), e4.Find());
        }


        //         [Fact]
        //         public void TestEgraphDump()
        //         {
        //             Expr a = 1 + 2;
        //             Expr b = 1 << 2;
        //             Expr c = a * b;
        //             EGraph graph = new EGraph();
        //             graph.Add(c);
        //             EGraphPrinter.DumpEgraphAsDot(graph, "example.dot");
        //         }

        //         [Fact]
        //         public void TestEgraphDumpAddSame()
        //         {
        //             Expr a = (Const)1 * 2;
        //             Expr b = (Const)1 * 2;
        //             EGraph graph = new EGraph();
        //             graph.Add(a);
        //             graph.Add(b);
        //             EGraphPrinter.DumpEgraphAsDot(graph, "exampleAddSame.dot");
        //         }

        //         [Fact]
        //         public void TestEgraphDumpVieta()
        //         {
        //             Var a = new Var("a");
        //             Var b = new Var("b");
        //             Var c = new Var("c");
        //             Expr d = (-b + Sqrt(Pow(b, 2) - 4 * a * c)) / (2 * a);

        //             EGraph graph = new EGraph();
        //             graph.Add(d);
        //             EGraphPrinter.DumpEgraphAsDot(graph, "exampleVieta.dot");
        //         }

        //         [Fact]
        //         public void TestEgraphMerge()
        //         {
        //             EGraph graph = new EGraph();
        //             Var a = new Var("a");
        //             Expr b = a * 2 / 2;
        //             graph.Add(b, out var bId);
        //             EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge1.dot");
        //             Expr c = a * 1;
        //             graph.Add(c, out var cId);
        //             EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge2.dot");
        //             graph.Add(Exp(b) + 3, out var dId);
        //             EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge3.dot");

        //             graph.Merge(cId, bId);
        //             EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge4.dot");
        //             graph.ReBuild();
        //             EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge5.dot");
        //         }

        //         [Fact]
        //         public void TestEgraphMatch()
        //         {
        //             //Given
        //             Var x = new Var("x", new TensorType(DataTypes.Float32, Shape.Scalar));
        //             Var y = new Var("y", new TensorType(DataTypes.Float32, Shape.Scalar));
        //             Expr pattern = x + y;
        //             //When
        //             Expr e1 = (Expr)1.0f * 2 / 2;
        //             Expr e2 = e1 + 100.0f; /* will match */
        //             Expr e3 = e2 - 10 + 100; /* will match in subset */
        //             var g = new EGraph();
        //             g.Add(e3);
        //         }

        //         [Fact]
        //         public void TestRebuildCanonicalizeEclass()
        //         {
        //             // When (x*2)+1 match x<<1, the (x*2)<==(x<<1) will be merge, 
        //             // but after rebuid, (x*2) is not in worklist, 
        //             // so it's eclass in hashcon will not be update
        //             // should fix it.
        //             passOptions.SetName("EGraphTest/TestRebuildCanonicalizeEclass");
        //             var g = new EGraph();
        //             Var x = "x";
        //             g.Add(x * 2, out var e1);
        //             g.Add(x << 1, out var e2);
        //             g.Add(((x * 2) + 1) + 3, out var root);
        //             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "before"));
        //             g.Merge(e1, e2);
        //             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "merge"));
        //             g.ReBuild();
        //             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "rebuild"));
        //             foreach (var (enode, eclass) in g.HashCons)
        //             {
        //                 Assert.Equal(eclass.Find(), eclass.Find());
        //             }
        //         }

        //         [Fact]
        //         public void TestRebuildUpdateUsed()
        //         {
        //             passOptions.SetName("EGraphTest/TestRebuildUpdateUsed");
        //             var g = new EGraph();
        //             Var x = "x";
        //             var expr1 = x * 2;
        //             var expr2 = x << 1;
        //             var expr3 = x * 4;
        //             var expr4 = x * 2 * 2;
        //             var y = (x * 2) * (x * 4);
        //             g.Add(expr1, out var e1);
        //             g.Add(expr2, out var e2);
        //             g.Add(expr3, out var e3);
        //             g.Add(expr4, out var e4);
        //             g.Add(y, out var root);
        //             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "before"));
        //             g.Merge(e2, e1);
        //             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "merge_lhs"));
        //             g.ReBuild();
        //             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "rebuild_lhs"));
        //             g.Merge(e4, e3);
        //             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "merge_rhs"));
        //             g.ReBuild();
        //             EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "rebuild_rhs"));
        //             foreach (var (enode, eclass) in g.HashCons)
        //             {
        //                 foreach (var child in enode.Children)
        //                 {
        //                     Assert.Contains(enode, child.Used.Select(kv => kv.Item1));
        //                 }
        //             }
        //         }

    }
}
