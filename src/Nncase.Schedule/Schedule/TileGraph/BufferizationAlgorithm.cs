// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Affine;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Algorithms.ShortestPath;
using QuikGraph.Graphviz;

namespace Nncase.Schedule.TileGraph;

public enum BufferEdgeKind
{
    Inter,
    Outer,
}

public sealed class BufferGraph : TieredAdjacencyGraph<BufferIdentity, EquatableTaggedEdge<BufferIdentity, BufferEdgeKind>>
{
    public BufferGraph(int topLevel, [NotNull] AdjacencyGraph<BufferIdentity, EquatableTaggedEdge<BufferIdentity, BufferEdgeKind>> wrappedGraph)
        : base(wrappedGraph)
    {
        OpId = -1;
        Level = topLevel;
    }

    public BufferGraph([NotNull] BufferGraph parentGraph, int level, int opid)
        : base(parentGraph)
    {
        OpId = opid;
        Level = level;
    }

    public int Level { get; }

    public int OpId { get; }

    public override string ToString() => $"Op{OpId}@{Level}";
}

public sealed class BufferizationAlgorithm : AlgorithmBase<TieredTileGraph>
{
    public BufferizationAlgorithm(TieredTileGraph visitedGraph)
        : base(visitedGraph)
    {
        BufferGraphMemo = new();
    }

    public Dictionary<TieredTileGraph, BufferGraph> BufferGraphMemo { get; internal set; }

    protected override void InternalCompute()
    {
        Visit(VisitedGraph);
    }

    private void Visit(TieredTileGraph rootGraph)
    {
        if (!BufferGraphMemo.TryGetValue(rootGraph, out _))
        {
            var wrappedGraph = new AdjacencyGraph<BufferIdentity, EquatableTaggedEdge<BufferIdentity, BufferEdgeKind>>();
            var rootBufferGraph = new BufferGraph(rootGraph.Level, wrappedGraph);
            Visit(rootGraph, rootBufferGraph);
            foreach (var edge in rootGraph.Edges)
            {
                var source = new BufferIdentity(edge.Source, edge.Source.ReadAccesses.Length);
                var target = new BufferIdentity(edge.Target, edge.Tag);
                rootBufferGraph.AddEdge(new(source, target, BufferEdgeKind.Outer));
            }

            BufferGraphMemo.Add(rootGraph, rootBufferGraph);
        }
    }

    private void Visit(TieredTileGraph graph, BufferGraph bufferGraph)
    {
        if (graph.ClustersCount == 0)
        {
            foreach (var item in graph.Vertices)
            {
                var outBid = new BufferIdentity(item, item.ReadAccesses.Length);
                for (int i = 0; i < item.ReadAccesses.Length; i++)
                {
                    bufferGraph.AddVerticesAndEdge(new(new(item, i), outBid, BufferEdgeKind.Inter));
                }
            }
        }
        else
        {
            foreach (var childGraph in graph.Clusters.OfType<TieredTileGraph>())
            {
                if (!BufferGraphMemo.TryGetValue(graph, out _))
                {
                    var childBufferGraph = bufferGraph.CreateCluster<BufferGraph>(childGraph.Level, childGraph.OpId);
                    Visit(childGraph, childBufferGraph);
                    BufferGraphMemo.Add(childGraph, childBufferGraph);
                }
            }
        }
    }
}
