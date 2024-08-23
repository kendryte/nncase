// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Algorithms.ShortestPath;
using QuikGraph.Graphviz;

namespace Nncase.Schedule.TileGraph;

public sealed record MergePoint(TileGrid Consumer, TileGrid Producer, int Level)
{
    public override string ToString() => $"merge({Consumer},{Producer},{Level})";
}

public sealed class GraphMerger
{
    public GraphMerger(TileGrid opConsumer, TileGrid opProducer, int level)
    {
        ConsumerOp = opConsumer;
        ProducerOp = opProducer;
        TargetLevel = level;
        RootGraph = null!;
    }

    public TileGrid ConsumerOp { get; }

    public TileGrid ProducerOp { get; }

    public int TargetLevel { get; }

    public TieredTileGraph RootGraph { get; set; }

    public bool Visit(TieredTileGraph graph)
    {
        RootGraph = graph;
        return VisitRecursion(graph);
    }

    private bool TryMerge(TieredTileGraph graph)
    {
        if (!GatherSubGraphs(graph, out var producerGraph, out var consumerGraph))
        {
            return false;
        }

        if (!CheckSubGraphsDenpendence(producerGraph, consumerGraph))
        {
            return false;
        }

        System.Diagnostics.Trace.Assert(ReferenceEquals(producerGraph.Parent, consumerGraph.Parent));
        System.Diagnostics.Trace.Assert(producerGraph.Level.Equals(consumerGraph.Level));

        var commonAncestor = producerGraph.Parent!;

        // 1. find the dataflow graph
        // 1.1 find the directly connected opnode with producer op.
        var algo = new FloydWarshallAllShortestPathAlgorithm<TileGrid, EquatableTaggedEdge<TileGrid, int>>(RootGraph, (_) => 1.0f);
        if (!algo.TryGetPath(ProducerOp, ConsumerOp, out var dependencePath))
        {
            return false;
        }

        var relayOp = dependencePath.First().Target;

        // 1.2. build the dataflow graph from consumer graph -> sub graph -> relay node.
        ITileable consumerParent = consumerGraph;
        var relationChain = new List<ITileable>();
        while (consumerParent is TieredTileGraph tileGraph)
        {
            ITileable consumerChild = tileGraph.Clusters.OfType<TieredTileGraph>().Where(sg => sg.ContainsVertex(relayOp)).Cast<ITileable>().FirstOrDefault(relayOp);
            relationChain.Add(consumerChild);
            consumerParent = consumerChild;
        }

        // 1.3 build the domain relation betwwen relay op -> producer op.
        var readAccess = relayOp.ReadAccesses[dependencePath.First().Tag];
        var relation = readAccess * AffineUtility.Inverse(ProducerOp.WriteAccess, ProducerOp.DomainBounds.Select(Convert.ToInt64).ToArray());
        if (!relation.IsProjectedPermutation(true))
        {
            return false;
        }

        var domainRel = new DomainRelation(relayOp.OpId, ProducerOp.OpId, relation);

        // 1.4 apply domain relation until consumerGraph
        foreach (var mappable in relationChain.Reverse<ITileable>())
        {
            domainRel = mappable.DomainRelation.ApplyRange(domainRel);
        }

        // 4. merge producerGraph's subgrph into the consumerGraph.
        commonAncestor.RemoveCluster(producerGraph);
        if (producerGraph.ClustersCount == 0)
        {
            foreach (var vertex in producerGraph.Vertices)
            {
                vertex.DomainRelation = domainRel.ApplyRange(vertex.DomainRelation);
            }
        }
        else
        {
            foreach (var producerChild in producerGraph.Clusters.OfType<TieredTileGraph>())
            {
                producerChild.DomainRelation = domainRel.ApplyRange(producerChild.DomainRelation);
                consumerGraph.AddCluster(producerChild);
            }
        }

        consumerGraph.AddVertexRange(producerGraph.Vertices);
        return true;
    }

    private bool CheckSubGraphsDenpendence(TieredTileGraph producer, TieredTileGraph consumer)
    {
        // 1. ensure there is no dependence cycle between producer and consumer.
        var subGraphGraph = new AdjacencyGraph<TieredTileGraph, Edge<TieredTileGraph>>();
        foreach (var edge in RootGraph.Edges)
        {
            if (producer.ContainsVertex(edge.Source) && consumer.ContainsVertex(edge.Target))
            {
                subGraphGraph.AddVerticesAndEdge(new(producer, consumer));
            }
            else if (producer.ContainsVertex(edge.Target) && consumer.ContainsVertex(edge.Source))
            {
                subGraphGraph.AddVerticesAndEdge(new(consumer, producer));
            }
        }

#if false
        var graphviz = subGraphGraph.ToGraphviz(init => { init.FormatVertex += (_, arg) => arg.VertexFormat.Label = $"{arg.Vertex.OpId}@{arg.Vertex.Level}"; });
#endif

        bool hasCycles = false;
        bool hasDependence = false;
        var dfs = new QuikGraph.Algorithms.Search.EdgeDepthFirstSearchAlgorithm<TieredTileGraph, Edge<TieredTileGraph>>(subGraphGraph);
        dfs.BackEdge += (edge) =>
        {
            hasCycles = true;
        };

        dfs.TreeEdge += (edge) =>
        {
            if (ReferenceEquals(edge.Source, producer) && ReferenceEquals(edge.Target, consumer))
            {
                hasDependence = true;
            }
        };

        dfs.Compute();

        return hasDependence && !hasCycles;
    }

    private bool GatherSubGraphs(TieredTileGraph graph, [MaybeNullWhen(false)] out TieredTileGraph producer, [MaybeNullWhen(false)] out TieredTileGraph consumer)
    {
        producer = null!;
        consumer = null!;
        foreach (var s1 in graph.Clusters.OfType<TieredTileGraph>().Where(s => s.OpId == ProducerOp.OpId))
        {
            foreach (var s2 in graph.Clusters.OfType<TieredTileGraph>().Where(s => s.OpId == ConsumerOp.OpId))
            {
                if (s1.ContainsVertex(ProducerOp) && !s1.ContainsVertex(ConsumerOp) &&
                    !s2.ContainsVertex(ProducerOp) && s2.ContainsVertex(ConsumerOp))
                {
                    producer = s1;
                    consumer = s2;
                    return true;
                }
            }
        }

        return false;
    }

    private bool VisitRecursion(TieredTileGraph graph)
    {
        if (graph.Level == TargetLevel + 1)
        {
            return TryMerge(graph);
        }

        if (graph.Level <= TargetLevel)
        {
            return false;
        }

        foreach (var subGraph in graph.Clusters.OfType<TieredTileGraph>())
        {
            if (VisitRecursion(subGraph))
            {
                return true;
            }
        }

        return false;
    }
}
