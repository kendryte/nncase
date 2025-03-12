// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.Tests.TestFixture;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Graphviz;
using Xunit;

namespace Nncase.Tests.Graphs;

[AutoSetupTestMethod(InitSession = true)]
public class UnitTestCondensationGraphAlgorithm : TestClassBase
{
    [Fact]
    public void TestSimpleBidirectionalGraph()
    {
        var graph = GraphCase0();
        Dump(graph, "biGraph");
    }

    [Fact]
    public void TestMerge0()
    {
        var graph = GraphCase0();
        var condseAlgo = new CondensationGraphAlgorithm<int, Edge<int>>(graph);
        condseAlgo.Compute();
        condseAlgo.MergeTwoVertex(0, 1);
        condseAlgo.MergeTwoVertex(2, 3);

        Dump(condseAlgo.ClusteredGraph, "ClusteredGraph");
        Dump(condseAlgo.CondensedGraph, "CondensedGraph");
        Assert.Equal(3, condseAlgo.CondensedGraph.VertexCount);
        Assert.Equal(2, condseAlgo.CondensedGraph.OutDegree(condseAlgo.VertexMap[0]));
        Assert.Equal(2, condseAlgo.CondensedGraph.InDegree(condseAlgo.VertexMap[2]));
        Assert.Equal(condseAlgo.VertexMap[0], condseAlgo.VertexMap[1]);
        Assert.Equal(condseAlgo.VertexMap[2], condseAlgo.VertexMap[3]);
    }

    [Fact]
    public void TestMerge1()
    {
        var graph = new BidirectionalGraph<int, Edge<int>>(false);
        graph.AddVerticesAndEdge(new(0, 1));
        graph.AddVerticesAndEdge(new(2, 1));
        graph.AddVerticesAndEdge(new(1, 3));
        graph.AddVerticesAndEdge(new(4, 3));
        var condseAlgo = new CondensationGraphAlgorithm<int, Edge<int>>(graph);
        condseAlgo.Compute();
        condseAlgo.MergeTwoVertex(0, 1);
        condseAlgo.MergeTwoVertex(2, 1);
        condseAlgo.MergeTwoVertex(3, 1);

        Dump(condseAlgo.ClusteredGraph, "ClusteredGraph");
        Dump(condseAlgo.CondensedGraph, "CondensedGraph");
        Assert.Equal(2, condseAlgo.CondensedGraph.VertexCount);
        Assert.Equal(4, condseAlgo.VertexMap[0].VertexCount);
        Assert.Equal(1, condseAlgo.VertexMap[4].VertexCount);
    }

    [Fact]
    public void TestSplit0()
    {
        var graph = new BidirectionalGraph<int, Edge<int>>(false);
        graph.AddVerticesAndEdge(new(0, 1));
        graph.AddVerticesAndEdge(new(2, 1));
        graph.AddVerticesAndEdge(new(1, 3));
        graph.AddVerticesAndEdge(new(4, 3));
        var condseAlgo = new CondensationGraphAlgorithm<int, Edge<int>>(graph);
        condseAlgo.Compute();
        condseAlgo.MergeTwoVertex(0, 1);
        condseAlgo.MergeTwoVertex(2, 1);
        condseAlgo.MergeTwoVertex(3, 4);
        var info = condseAlgo.MergeTwoVertex(1, 3);

        Dump(condseAlgo.ClusteredGraph, "MergeClusteredGraph");
        Dump(condseAlgo.CondensedGraph, "MergeCondensedGraph");
        Assert.Equal(1, condseAlgo.CondensedGraph.VertexCount);

        condseAlgo.SplitTwoVertex(info);
        Dump(condseAlgo.ClusteredGraph, "SplitClusteredGraph");
        Dump(condseAlgo.CondensedGraph, "SplitCondensedGraph");
        Assert.Equal(2, condseAlgo.CondensedGraph.VertexCount);
        Assert.Equal(3, condseAlgo.VertexMap[0].VertexCount);
        Assert.Equal(2, condseAlgo.VertexMap[4].VertexCount);
    }

    [Fact]
    public void TestCondensation0()
    {
        var graph = GraphCase0();
        var condseAlgo = new CondensationGraphAlgorithm<int, Edge<int>>(graph);
        condseAlgo.IsEdgeCompatible += (s, arg) =>
        {
            if (arg.Edge is { Source: 1, Target: 3 })
            {
                return true;
            }

            return false;
        };
        condseAlgo.Compute();

        Dump(condseAlgo.ClusteredGraph, "ClusteredGraph");
        Dump(condseAlgo.CondensedGraph, "CondensedGraph");
    }

    [Fact]
    public void TestCondensation1()
    {
        var graph = GraphCase0();
        var condseAlgo = new CondensationGraphAlgorithm<int, Edge<int>>(graph);
        condseAlgo.IsEdgeCompatible += (s, arg) =>
        {
            if (arg.Edge is { Source: 1, Target: 3 } or { Source: 0, Target: 1 })
            {
                return true;
            }

            return false;
        };
        condseAlgo.Compute();

        Dump(condseAlgo.ClusteredGraph, "ClusteredGraph");
        Dump(condseAlgo.CondensedGraph, "CondensedGraph");

        Assert.False(condseAlgo.CondensedGraph.IsDirectedAcyclicGraph());
    }

    [Fact]
    public void TestCondensation2()
    {
        var graph = GraphCase0();
        var condseAlgo = new CondensationGraphAlgorithm<int, Edge<int>>(graph);
        condseAlgo.IsEdgeCompatible += (s, arg) =>
        {
            if (arg.Edge is { Source: 1, Target: 3 } or { Source: 0, Target: 1 })
            {
                return true;
            }

            return false;
        };
        condseAlgo.IsGraphCompatible += (s, arg) =>
        {
            return s.CondensedGraph.IsDirectedAcyclicGraph();
        };
        condseAlgo.Compute();

        Dump(condseAlgo.ClusteredGraph, "ClusteredGraph");
        Dump(condseAlgo.CondensedGraph, "CondensedGraph");

        // NOTE actually we can't merge 1,3 and 0,1 simultaneously, but according to the dfs order, finally the 0,1 will be merge.
        Assert.Equal(4, condseAlgo.CondensedGraph.VertexCount);
    }

    private BidirectionalGraph<int, Edge<int>> GraphCase0()
    {
        var biGraph = new BidirectionalGraph<int, Edge<int>>(false);
        biGraph.AddVerticesAndEdge(new(0, 1));
        biGraph.AddVerticesAndEdge(new(0, 2));
        biGraph.AddVerticesAndEdge(new(1, 3));
        biGraph.AddVerticesAndEdge(new(2, 3));
        biGraph.AddVerticesAndEdge(new(3, 4));
        return biGraph;
    }

    private void Dump<TVertex, TEdge>(IEdgeListGraph<TVertex, TEdge> graph, string name)
        where TEdge : IEdge<TVertex>
    {
#if DEBUG
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"{name}.dot"))
        {
            using (var writer = new StreamWriter(stream))
            {
                writer.Write(graph.ToGraphviz());
            }
        }
#endif
    }
}
