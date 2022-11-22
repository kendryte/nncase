using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Transform;
using Xunit;
using static Nncase.IR.F.Math;

namespace Nncase.Tests.EGraphTest;

public class UnitTestEGraph : TestFixture.UnitTestFixtrue
{
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
        EGraphPrinter.DumpEgraphAsDot(egraph, $"{nameof(TestENodeCallHashEGraph)}.dot");
    }

    [Fact]
    public void TestEgraphDump()
    {
        Expr a = (Const)1 + 2;
        Expr b = (Const)1 << 2;
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
        Assert.Equal(4, graph.Nodes.Count());
        EGraphPrinter.DumpEgraphAsDot(graph, "exampleAddSame.dot");
    }

    [Fact]
    public void TestEgraphDumpVieta()
    {
        Var a = new Var("a");
        Var b = new Var("b");
        Var c = new Var("c");
        Expr d = (-b + Sqrt(Pow(b, 2) - (4 * a * c))) / (2 * a);

        EGraph graph = new EGraph();
        graph.Add(d);
        Assert.Equal(21, graph.Nodes.Count());
        EGraphPrinter.DumpEgraphAsDot(graph, "exampleVieta.dot");
    }

    [Fact]
    public void TestEgraphMerge()
    {
        EGraph graph = new EGraph();
        Var a = new Var("a");
        Expr b = a * 2 / 2;
        var bId = graph.Add(b);
        EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge1.dot");

        Expr c = a * 1;
        var cId = graph.Add(c);
        EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge2.dot");

        var dId = graph.Add(Exp(b) + 3);
        var eId = graph.Add(Exp(c) + 3);
        EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge3.dot");

        graph.Union(cId, bId);
        Assert.NotStrictEqual(dId.Find(), eId.Find());
        EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge4.dot");

        graph.Rebuild();
        Assert.StrictEqual(bId.Find(), cId.Find());
        Assert.NotStrictEqual(dId.Find(), cId.Find());
        Assert.StrictEqual(dId.Find(), eId.Find());
        EGraphPrinter.DumpEgraphAsDot(graph, "exampleMerge5.dot");
    }

    [Fact]
    public void TestEgraphMatch()
    {
        //Given
        Var x = new Var("x", new TensorType(DataTypes.Float32, Shape.Scalar));
        Var y = new Var("y", new TensorType(DataTypes.Float32, Shape.Scalar));
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
        var passOptions = GetPassOptions();
        var g = new EGraph();
        Var x = "x";
        var e1 = g.Add(x * 2);
        var e2 = g.Add(x << 1);
        var e3 = g.Add((x * 2) + 1 + 3);
        var e4 = g.Add((x << 1) + 1 + 3);
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "before"));

        g.Union(e1, e2);
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "merge"));

        g.Rebuild();
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "rebuild"));

        Assert.StrictEqual(e3.Find(), e4.Find());
    }

    [Fact]
    public void TestRebuildUpdateUsed()
    {
        var passOptions = GetPassOptions();
        var g = new EGraph();
        Var x = "x";
        var e1 = g.Add(x * 2);
        var e2 = g.Add(x << 1);
        var e3 = g.Add(x * 4);
        var e4 = g.Add(x * 2 * 2);
        var e5 = g.Add((x * 2) * (x * 4));
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "before"));
        g.Union(e2, e1);
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "merge_lhs"));
        g.Rebuild();
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "rebuild_lhs"));
        g.Union(e4, e3);
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "merge_rhs"));
        g.Rebuild();
        EGraphPrinter.DumpEgraphAsDot(g, Path.Combine(passOptions.PassDumpDir, "rebuild_rhs"));
        foreach (var enode in g.Nodes)
        {
            foreach (var child in enode.Children)
            {
                Assert.Contains(enode, child.Used);
            }
        }
    }

}
