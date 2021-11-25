using System;
using Xunit;
using Nncase.Transform;
using Nncase.IR;
using static Nncase.IR.F.Math;
using System.Collections.Generic;

namespace Nncase.Tests
{
    public class UnitTestEGraph
    {
        [Fact]
        public void TestConstEqual()
        {
            Const a = 2;
            Const b = 2;
            var d = new Dictionary<Expr, int>();
            Console.WriteLine(a.Data == b.Data);
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

    }
}
