// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Schedule.TileGraph;
using QuikGraph;
using QuikGraph.Graphviz;
using Xunit;

namespace Nncase.Tests.AffineTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestTileGraph : TestClassBase
{
    public static readonly TheoryData<Func<Function>, int> BuildTileGraphDatas = new()
    {
        { FunctionSamples.Get1, 0 },
        { FunctionSamples.Get2, 1 },
        { FunctionSamples.Get3, 2 },
    };

    public static readonly TheoryData<Func<Function>, (IntMergePoint, bool)[], Action<TieredTileGraph>, int> MergeTileGraphDatas = new()
    {
        { FunctionSamples.Get1, new (IntMergePoint, bool)[] { (new(2, 1, 2), true), (new(2, 0, 2), true), (new(2, 0, 1), false), (new(1, 0, 1), true) }, MergeTileGraphChecker0, 0 },
        { FunctionSamples.Get1, new (IntMergePoint, bool)[] { (new(1, 0, 2), true), (new(2, 0, 2), false), (new(2, 1, 2), true), }, MergeTileGraphCheckerDefault, 1 },
        { FunctionSamples.Get1PackMN, new (IntMergePoint, bool)[] { (new(2, 0, 2), true), (new(2, 1, 2), true), (new(2, 0, 1), true), (new(2, 1, 1), true), (new(3, 2, 2), true), (new(5, 4, 2), true) }, MergeTileGraphChecker2, 2 },
    };

    [Fact]
    public void TestClusteredGraph()
    {
        string a = "a", b = "b", c = "c", d = "d", e = "e", f = "f";

        var g0 = new AdjacencyGraph<string, Edge<string>>();
        var g = new ClusteredAdjacencyGraph<string, Edge<string>>(g0);
        var g1 = g.AddCluster();
        var g2 = g.AddCluster();

        g1.AddVerticesAndEdge(new(e, f));
        g1.AddVerticesAndEdge(new(c, f));

        Assert.Equal(3, g1.VertexCount);

        g2.AddVerticesAndEdge(new(a, b));

        Assert.Equal(2, g2.VertexCount);

        Assert.Equal(5, g0.VertexCount);

        g.AddEdge(new(e, b));
        g.AddEdge(new(b, c));

        g.AddVertex(d);
        g.AddVerticesAndEdge(new(b, d));
        g.AddVerticesAndEdge(new(f, d));

#if DEBUG
        using (var file = Dumpper.OpenFile("g.dot"))
        {
            using var writer = new StreamWriter(file);
            writer.Write(g.ToGraphviz(algorithm =>
            {
                algorithm.FormatVertex += (_, args) => args.VertexFormat.Label = args.Vertex;
                algorithm.FormatCluster += (_, args) => args.GraphFormat.Label = "f";
            }));
        }
#endif

        // build a graph for subgraphs.
        var cg = new AdjacencyGraph<ClusteredAdjacencyGraph<string, Edge<string>>, Edge<ClusteredAdjacencyGraph<string, Edge<string>>>>();
        foreach (var subGraph in g.Clusters.OfType<ClusteredAdjacencyGraph<string, Edge<string>>>())
        {
            cg.AddVertex(subGraph);
        }

        foreach (var edge in g.Edges)
        {
            foreach (var source in cg.Vertices)
            {
                foreach (var target in cg.Vertices.Where(v => v != source))
                {
                    if (source.ContainsVertex(edge.Source) && target.ContainsVertex(edge.Target))
                    {
                        cg.AddEdge(new(source, target));
                    }
                }
            }
        }

#if DEBUG
        using (var file = Dumpper.OpenFile("cg.dot"))
        {
            using var writer = new StreamWriter(file);
            writer.Write(cg.ToGraphviz(algorithm =>
            {
            }));
        }
#endif
    }

    [Fact]
    public void TestClusteredGraphAsBufferGraph()
    {
        var g = new AdjacencyGraph<string, Edge<string>>();
        var cg = new ClusteredAdjacencyGraph<string, Edge<string>>(g);
        var cg0 = cg.AddCluster();
        var cg0_0 = cg0.AddCluster();
        var cg0_1 = cg0.AddCluster();

        cg0_0.AddVerticesAndEdge(new("op0_in0", "op0_out"));
        cg0_1.AddVerticesAndEdge(new("op1_in0", "op1_out"));
        cg0_1.AddVerticesAndEdge(new("op1_in1", "op1_out"));
        cg0.AddEdge(new("op0_out", "op1_in0"));

        var cg1 = cg.AddCluster();
        var cg1_0 = cg1.AddCluster();
        cg1_0.AddVerticesAndEdge(new("op2_in0", "op2_out"));

        cg.AddEdge(new("op1_out", "op2_in0"));

        var nameMap = new Dictionary<IVertexAndEdgeListGraph<string, Edge<string>>, string>() {
            { cg0, "cg0" },
            { cg0_0, "cg0_0" },
            { cg0_1, "cg0_1" },
            { cg1, "cg1" },
            { cg1_0, "cg1_0" },
        };

#if DEBUG
        using (var file = Dumpper.OpenFile("g.dot"))
        {
            using var writer = new StreamWriter(file);
            writer.Write(cg.ToGraphviz(algorithm =>
            {
                algorithm.FormatVertex += (_, args) => args.VertexFormat.Label = args.Vertex;
                algorithm.FormatCluster += (_, args) => args.GraphFormat.Label = nameMap[args.Cluster];
            }));
        }
#endif
    }

    [Fact]
    public void TestRemoveAndAddSubGraph()
    {
        AdjacencyGraph<string, Edge<string>> root = new();
        TieredAdjacencyGraph<string, Edge<string>> cg = new(root);
        var cg0 = cg.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();
        var cg00 = cg0.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();
        cg00.AddVertex("a");
        var cg1 = cg.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();
        var cg11 = cg1.CreateCluster<TieredAdjacencyGraph<string, Edge<string>>>();
        cg11.AddVertex("b");
        root.AddEdge(new("a", "b"));

        var cnames = new Dictionary<IVertexAndEdgeListGraph<string, Edge<string>>, string>()
        {
            { cg, "cg" },
            { cg0, "cg0" },
            { cg00, "cg00" },
            { cg1, "cg1" },
            { cg11, "cg11" },
        };

        void Dump(TieredAdjacencyGraph<string, Edge<string>> graph, string name)
        {
            using (var file = Dumpper.OpenFile($"{name}.dot"))
            {
                using var writer = new StreamWriter(file);
                writer.Write(graph.ToGraphviz(algorithm =>
                {
                    algorithm.FormatVertex += (_, args) => args.VertexFormat.Label = args.Vertex;
                    algorithm.FormatCluster += (_, args) => args.GraphFormat.Label = cnames[args.Cluster];
                }));
            }
        }

#if DEBUG
        Dump(cg, "cg");
#endif

        Assert.Equal(2, root.VertexCount);
        Assert.Equal(2, cg.VertexCount);
        Assert.Equal(1, cg0.VertexCount);
        Assert.Equal(1, cg1.VertexCount);

        void MergeSubGraph(TieredAdjacencyGraph<string, Edge<string>> source, TieredAdjacencyGraph<string, Edge<string>> target)
        {
            var parent = target.Parent!;

            // try move cg00 into cg1
            parent.RemoveCluster(source);
            foreach (var sourceChild in source.Clusters.OfType<TieredAdjacencyGraph<string, Edge<string>>>())
            {
                target.AddCluster(sourceChild);
            }

            target.AddVertexRange(source.Vertices);
        }

        MergeSubGraph(cg0, cg1);

#if DEBUG
        Dump(cg, "cg_merge_cg0_to_cg1");
#endif

        Assert.Equal(2, root.VertexCount); // keep the root vertex not change.
        Assert.Equal(2, cg.VertexCount);
        Assert.Equal(1, cg.ClustersCount);
        Assert.Equal(2, cg1.ClustersCount);
        Assert.Equal(2, cg1.VertexCount);
        Assert.Equal(1, root.EdgeCount);
        Assert.Equal(1, cg.EdgeCount);

        MergeSubGraph(cg11, cg00);

#if DEBUG
        Dump(cg, "cg_merge_cg11_to_cg00");
#endif

        Assert.Equal(2, root.VertexCount);
        Assert.Equal(2, cg.VertexCount);
        Assert.Equal(1, cg.ClustersCount);
        Assert.Equal(1, cg1.ClustersCount);
        Assert.Equal(0, cg00.ClustersCount);
        Assert.Equal(2, cg00.VertexCount);
        Assert.Equal(1, root.EdgeCount);
        Assert.Equal(1, cg.EdgeCount);
    }

    [Theory]
    [MemberData(nameof(BuildTileGraphDatas))]
    public void TestBuildTileGraph(Func<Function> functor, int count)
    {
        var func = functor();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerPack(), new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), new Passes.Rules.CPU.Affine.LowerBinary() }, new());
#if DEBUG
        Dumpper.DumpIR(post, $"post{count}");
#endif

        var builder = new GraphBuilder(2);
        builder.Visit(post);
        var graph = builder.RootGraph;
#if DEBUG
        graph.Dump($"g{count}");
#endif

        Assert.Equal(3, graph.Level);
    }

    [Theory]
    [MemberData(nameof(MergeTileGraphDatas))]
    public void TestMergeTileGraph(Func<Function> functor, (IntMergePoint, bool)[] mergePoints, Action<TieredTileGraph> checker, int count)
    {
        var func = functor();
        var post = CompilerServices.Rewrite(func, new IRewriteRule[] { new Passes.Rules.CPU.Affine.LowerPack(), new Passes.Rules.CPU.Affine.LowerUnary(), new Passes.Rules.CPU.Affine.LowerMatmul(), new Passes.Rules.CPU.Affine.LowerBinary() }, new());
#if DEBUG
        Dumpper.DumpIR(post, $"post{count}");
#endif

        var builder = new GraphBuilder(2);
        builder.Visit(post);
        var tileGraph = builder.RootGraph;
#if DEBUG
        tileGraph.Dump($"g{count}");
#endif

        for (int i = 0; i < mergePoints.Length; i++)
        {
            var (point, excepted) = mergePoints[i];
            Assert.Equal(excepted, tileGraph.Merge(new(tileGraph.Vertices.Skip(point.Consumer).First(), tileGraph.Vertices.Skip(point.Producer).First(), point.Level)));
#if DEBUG
            if (excepted)
            {
                tileGraph.Dump($"g{count}_m{i}");
            }
#endif
        }

        checker(tileGraph);
    }

    private static void MergeTileGraphCheckerDefault(TieredTileGraph tileGraph)
    {
    }

    private static void MergeTileGraphChecker0(TieredTileGraph tileGraph)
    {
        tileGraph.Walk(g =>
        {
            if (g is TieredTileGraph { Level: 1, OpId: 1 } g1)
            {
                Assert.Equal(2, g1.VertexCount);
                foreach (var op in g1.Vertices.Where(v => v.OpId == 0))
                {
                    Assert.Equal(1, op.DomainRelation.DomainOp);
                    Assert.Equal(0, op.DomainRelation.RangeOp);
                }
            }
        });
    }

    private static void MergeTileGraphChecker2(TieredTileGraph tileGraph)
    {
        // (new(2, 0, 2), true), (new(2, 1, 2), true), (new(2, 0, 1), true), (new(2, 1, 1), true), (new(3, 2, 2), true), (new(5, 4, 2), true)
        tileGraph.Walk(g =>
        {
            if (g is TieredTileGraph { Level: 2, OpId: 5 } g1)
            {
                Assert.Equal(2, g1.VertexCount);
                Assert.Equal(2, g1.ClustersCount);
                foreach (var item in g1.Clusters.OfType<TieredTileGraph>())
                {
                    Assert.Equal(5, item.DomainRelation.DomainOp);
                    Assert.Equal(item.OpId, item.DomainRelation.RangeOp);
                }
            }

            if (g is TieredTileGraph { Level: 2, OpId: 2 } g2)
            {
                Assert.Equal(3, g2.VertexCount);
                Assert.Equal(1, g2.ClustersCount);
            }

            if (g is TieredTileGraph { Level: 1, OpId: 2 } g3)
            {
                Assert.Equal(3, g3.VertexCount);
                Assert.Equal(0, g3.ClustersCount);
                foreach (var item in g3.Vertices)
                {
                    Assert.Equal(2, item.DomainRelation.DomainOp);
                    Assert.Equal(item.OpId, item.DomainRelation.RangeOp);
                }
            }
        });
    }

    public sealed record IntMergePoint(int Consumer, int Producer, int Level)
    {
        public override string ToString() => $"merge({Consumer},{Producer},{Level})";
    }
}
