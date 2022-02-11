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

namespace Nncase.Tests
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
            string dumpDir = Path.Combine(GetThisFilePath(), "..", "..", "..", "tests_output");
            dumpDir = Path.GetFullPath(dumpDir);
            Directory.CreateDirectory(dumpDir);
            passOptions = new RunPassOptions(null, 3, dumpDir);
        }

        [Fact]
        public void TestConstEqual()
        {
            Const a = 2;
            Const b = 2;
            var d = new Dictionary<Expr, int>();
            Console.WriteLine(a == b);
            d.Add(a, 1);
            Console.WriteLine(d.TryGetValue(b, out var res));
        }

        [Fact]
        public void TestExprEqual()
        {
            Call a = Mul(1, 12);
            Call b = Mul(2, 3);
            EGraph graph = new EGraph();
            graph.Add(a * b);
            var d = new Dictionary<Expr, int>();
            Console.WriteLine(a.Target == b.Target);
            d.Add(a.Target, 1);
            Console.WriteLine(d.TryGetValue(b.Target, out var res));

            EGraphPrinter.DumpEgraphAsDot(graph, "exampleEqual.dot");
        }

        [Fact]
        public void TestAddExpr()
        {
            Expr a = 1 + IR.F.Math.Exp(2);
            EGraph graph = new EGraph();
            graph.Add(a);
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
            var node1 = new ENode((Const)1, new EClass[] { });
            var node2 = new ENode((Const)1, new EClass[] { });
            var node3 = new ENode((Const)11, new EClass[] { });
            Assert.Equal(node1, node2);
            Assert.Equal(node1.GetHashCode(), node2.GetHashCode());
            Assert.NotEqual(node1, node3);
            Assert.NotEqual(node1.GetHashCode(), node3.GetHashCode());
        }

        [Fact]
        public void TestENodeOpHash()
        {
            var node1 = new ENode((Op)new Binary(BinaryOp.Add), new EClass[] { });
            var node2 = new ENode((Op)new Binary(BinaryOp.Add), new EClass[] { });
            var node3 = new ENode((Op)new Unary(UnaryOp.Abs), new EClass[] { });
            Assert.Equal(node1, node2);
            Assert.Equal(node1.GetHashCode(), node2.GetHashCode());
            Assert.NotEqual(node1, node3);
            Assert.NotEqual(node1.GetHashCode(), node3.GetHashCode());
        }

        [Fact]
        public void TestENodeCallHash()
        { // when the expr have same eclass args, but their expr is not
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
            var call2 = Binary(BinaryOp.Add, ((Const)1 + 3), ((Const)2 + 2));
            egraph.Add((Const)4, out var e1);
            egraph.Add(((Const)1 + 3), out var e2);
            egraph.Add(((Const)2 + 2), out var e3);
            egraph.Add(call1, out var e4);
            egraph.Merge(e1, e2);
            egraph.Merge(e1, e3);
            egraph.ReBuild();
            egraph.Add(call2, out var e5);
            Assert.Equal(e5.Find(), e4.Find());
        }


        [Fact]
        public void TestEgraphDump()
        {
            Expr a = 1 + 2;
            Expr b = 1 << 2;
            Expr c = a * b;
            EGraph graph = new EGraph();
            graph.Add(c);
            EGraphPrinter.DumpEgraphAsDot(graph, "example.dot");
        }

        [Fact]
        public void TestEgraphDumpAddSame()
        {
            Expr a = (Const)1 * 2;
            Expr b = (Const)1 * 2;
            EGraph graph = new EGraph();
            graph.Add(a);
            graph.Add(b);
            EGraphPrinter.DumpEgraphAsDot(graph, "exampleAddSame.dot");
        }

        [Fact]
        public void TestEgraphDumpVieta()
        {
            Var a = new Var("a");
            Var b = new Var("b");
            Var c = new Var("c");
            Expr d = (-b + Sqrt(Pow(b, 2) - 4 * a * c)) / (2 * a);

            EGraph graph = new EGraph();
            graph.Add(d);
            EGraphPrinter.DumpEgraphAsDot(graph, "exampleVieta.dot");
        }

        [Fact]
        public void TestEgraphMerge()
        {
            EGraph graph = new EGraph();
            Var a = new Var("a");
            Expr b = a * 2 / 2;
            graph.Add(b, out var bId);
            EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge1.dot");
            Expr c = a * 1;
            graph.Add(c, out var cId);
            EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge2.dot");
            graph.Add(Exp(b) + 3, out var dId);
            EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge3.dot");

            graph.Merge(cId, bId);
            EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge4.dot");
            graph.ReBuild();
            EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge5.dot");
        }

        [Fact]
        public void TestEgraphMatch()
        {
            //Given
            Var x = new Var("x", new TensorType(DataType.Float32, Shape.Scalar));
            Var y = new Var("y", new TensorType(DataType.Float32, Shape.Scalar));
            Expr pattern = x + y;
            //When
            Expr e1 = (Expr)1.0f * 2 / 2;
            Expr e2 = e1 + 100.0f; /* will match */
            Expr e3 = e2 - 10 + 100; /* will match in subset */
            var g = new EGraph();
            g.Add(e3);
        }

        [Fact]
        public void TestRebuildCanonicalizeEclass()
        {
            // When (x*2)+1 match x<<1, the (x*2)<==(x<<1) will be merge, 
            // but after rebuid, (x*2) is not in worklist, 
            // so it's eclass in hashcon will not be update
            // should fix it.
            passOptions.SetName("EGraphTest/TestRebuildCanonicalizeEclass");
            var g = new EGraph();
            Var x = "x";
            g.Add(x * 2, out var e1);
            g.Add(x << 1, out var e2);
            g.Add(((x * 2) + 1) + 3, out var root);
            EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.FullDumpDir, "before"));
            g.Merge(e1, e2);
            EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.FullDumpDir, "merge"));
            g.ReBuild();
            EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.FullDumpDir, "rebuild"));
            foreach (var (enode, eclass) in g.HashCons)
            {
                Assert.Equal(eclass.Find(), eclass.Find());
            }
        }

        [Fact]
        public void TestRebuildUpdateUsed()
        {
            passOptions.SetName("EGraphTest/TestRebuildUpdateUsed");
            var g = new EGraph();
            Var x = "x";
            var expr1 = x * 2;
            var expr2 = x << 1;
            var expr3 = x * 4;
            var expr4 = x * 2 * 2;
            var y = (x * 2) * (x * 4);
            g.Add(expr1, out var e1);
            g.Add(expr2, out var e2);
            g.Add(expr3, out var e3);
            g.Add(expr4, out var e4);
            g.Add(y, out var root);
            EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.FullDumpDir, "before"));
            g.Merge(e2, e1);
            EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.FullDumpDir, "merge_lhs"));
            g.ReBuild();
            EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.FullDumpDir, "rebuild_lhs"));
            g.Merge(e4, e3);
            EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.FullDumpDir, "merge_rhs"));
            g.ReBuild();
            EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.FullDumpDir, "rebuild_rhs"));
            foreach (var (enode, eclass) in g.HashCons)
            {
                foreach (var child in enode.Children)
                {
                    Assert.Contains(enode, child.Used.Select(kv => kv.Item1));
                }
            }
        }

    }
}
