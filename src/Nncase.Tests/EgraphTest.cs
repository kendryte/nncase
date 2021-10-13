using System;
using Xunit;
using Nncase.Transform;
using Nncase.IR;
using GiGraph.Dot.Extensions;

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
            var dot = graph.Dump();
            dot.SaveToFile("example.dot");
        }
    }
}
