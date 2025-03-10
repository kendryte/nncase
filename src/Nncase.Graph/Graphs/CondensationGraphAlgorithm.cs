// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using QuikGraph;
using QuikGraph.Algorithms;

namespace Nncase.Graphs;

public delegate bool IsEdgeCompatibleEventHandler<TVertex, TEdge>(CondensationGraphAlgorithm<TVertex, TEdge> sender, IsEdgeCompatibleEventArgs<TVertex, TEdge> args)
    where TVertex : notnull
    where TEdge : class, IEdge<TVertex>;

public delegate bool IsGraphCompatibleEventHandler<TVertex, TEdge>(CondensationGraphAlgorithm<TVertex, TEdge> sender, IsGraphCompatibleEventArgs<TVertex, TEdge> args)
    where TVertex : notnull
    where TEdge : class, IEdge<TVertex>;

public sealed class IsGraphCompatibleEventArgs<TVertex, TEdge> : EventArgs
 where TEdge : IEdge<TVertex>
{
    public IsGraphCompatibleEventArgs(TEdge edge)
    {
        Edge = edge;
    }

    public TEdge Edge { get; }
}

public sealed class IsEdgeCompatibleEventArgs<TVertex, TEdge> : EventArgs
 where TEdge : IEdge<TVertex>
{
    public IsEdgeCompatibleEventArgs(TEdge edge)
    {
        Edge = edge;
    }

    public TEdge Edge { get; }
}

/// <summary>
/// Algorithm that condensate a graph with custom.
/// </summary>
/// <typeparam name="TVertex">Vertex type.</typeparam>
/// <typeparam name="TEdge">Edge type.</typeparam>
public sealed class CondensationGraphAlgorithm<TVertex, TEdge> : AlgorithmBase<IEdgeListAndIncidenceGraph<TVertex, TEdge>>
    where TEdge : class, IEdge<TVertex>
    where TVertex : notnull
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CondensationGraphAlgorithm{TVertex,TEdge}"/> class.
    /// </summary>
    /// <param name="visitedGraph">Graph to visit.</param>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="visitedGraph"/> is <see langword="null"/>.</exception>
    public CondensationGraphAlgorithm(IEdgeListAndIncidenceGraph<TVertex, TEdge> visitedGraph)
        : base(visitedGraph)
    {
        CondensedGraph = new(true);
        WrappedGraph = new(false);
        ClusteredGraph = new(WrappedGraph);
        VertexMap = new Dictionary<TVertex, ClusteredBidirectionalGraph<TVertex, TEdge>>();
        EdgeMap = new(ReferenceEqualityComparer.Instance);
    }

    public event IsEdgeCompatibleEventHandler<TVertex, TEdge>? IsEdgeCompatible;

    public event IsGraphCompatibleEventHandler<TVertex, TEdge>? IsGraphCompatible;

    public BidirectionalGraph<ClusteredBidirectionalGraph<TVertex, TEdge>, Edge<ClusteredBidirectionalGraph<TVertex, TEdge>>> CondensedGraph { get; }

    public BidirectionalGraph<TVertex, TEdge> WrappedGraph { get; }

    public ClusteredBidirectionalGraph<TVertex, TEdge> ClusteredGraph { get; }

    public Dictionary<TVertex, ClusteredBidirectionalGraph<TVertex, TEdge>> VertexMap { get; }

    public Dictionary<TEdge, Edge<ClusteredBidirectionalGraph<TVertex, TEdge>>> EdgeMap { get; }

    public MergeInfo MergeTwoVertex(TVertex source, TVertex target)
    {
        var sourceCluster = VertexMap[source];
        var targetCluster = VertexMap[target];
        var mergedCluster = ClusteredGraph.AddCluster();

        // remove cluster
        var (sourceRemovedVertices, sourceRemovedEdges) = RemoveCluster(sourceCluster);
        var (targetRemovedVertices, targetRemovedEdges) = RemoveCluster(targetCluster);

        // add merged cluster
        foreach (var removedVertex in sourceRemovedVertices.Concat(targetRemovedVertices))
        {
            mergedCluster.AddVertex(removedVertex);
            VertexMap.Add(removedVertex, mergedCluster);
        }

        CondensedGraph.AddVertex(mergedCluster);
        foreach (var removedEdge in sourceRemovedEdges.Concat(targetRemovedEdges))
        {
            ReAddEdges(removedEdge);
        }

        return new(sourceRemovedVertices, sourceRemovedEdges, targetRemovedVertices, targetRemovedEdges, mergedCluster);
    }

    public void SplitTwoVertex(MergeInfo mergeInfo)
    {
        RemoveCluster(mergeInfo.MergedCluster);
        var sourceCluster = ClusteredGraph.AddCluster();
        foreach (var vertex in mergeInfo.SourceRemovedVertices)
        {
            sourceCluster.AddVertex(vertex);
            VertexMap.Add(vertex, sourceCluster);
        }

        var targetCluster = ClusteredGraph.AddCluster();
        foreach (var vertex in mergeInfo.TargetRemovedVertices)
        {
            targetCluster.AddVertex(vertex);
            VertexMap.Add(vertex, targetCluster);
        }

        CondensedGraph.AddVertex(sourceCluster);
        CondensedGraph.AddVertex(targetCluster);
        foreach (var removedEdge in mergeInfo.SourceRemovedEdges.Concat(mergeInfo.TargetRemovedEdges))
        {
            ReAddEdges(removedEdge);
        }
    }

    /// <inheritdoc />
    protected override void InternalCompute()
    {
        if (VisitedGraph.VertexCount == 0)
        {
            return;
        }

        // 1. add vertices and edges into graphs.
        foreach (var vertex in VisitedGraph.Vertices)
        {
            var cluster = ClusteredGraph.AddCluster();
            cluster.AddVertex(vertex);
            CondensedGraph.AddVertex(cluster);
            VertexMap.Add(vertex, cluster);
        }

        foreach (var edge in VisitedGraph.Edges)
        {
            var condensedEdge = new Edge<ClusteredBidirectionalGraph<TVertex, TEdge>>(VertexMap[edge.Source], VertexMap[edge.Target]);
            CondensedGraph.AddEdge(condensedEdge);
            ClusteredGraph.AddEdge(edge);
            EdgeMap.Add(edge, condensedEdge);
        }

        // 2. try to merge vertices.
        var dfsVisitEdge = new QuikGraph.Algorithms.Search.EdgeDepthFirstSearchAlgorithm<TVertex, TEdge>(VisitedGraph);
        dfsVisitEdge.InitializeEdge += (edge) =>
        {
            var compatible = IsEdgeCompatible?.Invoke(this, new(edge)) ?? false;
            if (!compatible)
            {
                return;
            }

            // modify the condensated graph and clustered graph.
            var mergeInfo = MergeTwoVertex(edge.Source, edge.Target);

            compatible = IsGraphCompatible?.Invoke(this, new(edge)) ?? true;

            if (!compatible)
            {
                SplitTwoVertex(mergeInfo);
            }
        };
        dfsVisitEdge.Compute();
    }

    private (List<TVertex> RemovedVertices, List<TEdge> RemovedEdges) RemoveCluster(ClusteredBidirectionalGraph<TVertex, TEdge> cluster)
    {
        var removedVertices = new List<TVertex>();
        var removedEdges = new List<TEdge>();
        EdgeAction<TVertex, TEdge> edgeEvent = (e) =>
        {
            CondensedGraph.RemoveEdge(EdgeMap[e]);
            EdgeMap.Remove(e);
            removedEdges.Add(e);
        };

        ClusteredGraph.EdgeRemoved += edgeEvent;
        foreach (var item in cluster.Vertices)
        {
            ClusteredGraph.RemoveVertex(item);
            VertexMap.Remove(item);
            removedVertices.Add(item);
        }

        ClusteredGraph.EdgeRemoved -= edgeEvent;
        ClusteredGraph.RemoveCluster(cluster);
        CondensedGraph.RemoveVertex(cluster);
        return (removedVertices, removedEdges);
    }

    private void ReAddEdges(TEdge removedEdge)
    {
        var condensedEdge = new Edge<ClusteredBidirectionalGraph<TVertex, TEdge>>(VertexMap[removedEdge.Source], VertexMap[removedEdge.Target]);
        if (!ReferenceEquals(condensedEdge.Source, condensedEdge.Target))
        {
            CondensedGraph.AddEdge(condensedEdge);
            ClusteredGraph.AddEdge(removedEdge);
        }
        else
        {
            condensedEdge.Source.AddEdge(removedEdge);
        }

        EdgeMap.Add(removedEdge, condensedEdge);
    }

    public record MergeInfo(List<TVertex> SourceRemovedVertices, List<TEdge> SourceRemovedEdges, List<TVertex> TargetRemovedVertices, List<TEdge> TargetRemovedEdges, ClusteredBidirectionalGraph<TVertex, TEdge> MergedCluster)
    {
    }
}
