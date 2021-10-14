using System;
using Xunit;
using Nncase.Transform;
using Nncase.IR;
using static Nncase.IR.F.Math;

namespace Nncase.Tests
{
    public class UnitTestEGraph
    {
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

    }
}
