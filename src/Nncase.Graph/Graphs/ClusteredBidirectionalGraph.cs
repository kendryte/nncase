// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
#if SUPPORTS_AGGRESSIVE_INLINING
using System.Runtime.CompilerServices;
#endif
using JetBrains.Annotations;
using QuikGraph;
using QuikGraph.Collections;

namespace Nncase.Graphs;

/// <summary>
/// Mutable clustered adjacency graph data structure.
/// </summary>
/// <typeparam name="TVertex">Vertex type.</typeparam>
/// <typeparam name="TEdge">Edge type.</typeparam>
#if SUPPORTS_SERIALIZATION
    [Serializable]
#endif
[DebuggerDisplay("VertexCount = {" + nameof(VertexCount) + "}, EdgeCount = {" + nameof(EdgeCount) + "}")]
public class ClusteredBidirectionalGraph<TVertex, TEdge>
    : IEdgeListAndIncidenceGraph<TVertex, TEdge>,
      IMutableBidirectionalGraph<TVertex, TEdge>,
      IClusteredGraph
    where TEdge : IEdge<TVertex>
{
    [NotNull]
    private readonly List<IClusteredGraph> _clusters = new List<IClusteredGraph>();

    /// <summary>
    /// Initializes a new instance of the <see cref="ClusteredBidirectionalGraph{TVertex,TEdge}"/> class.
    /// </summary>
    /// <param name="wrappedGraph">Graph to wrap.</param>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="wrappedGraph"/> is <see langword="null"/>.</exception>
    public ClusteredBidirectionalGraph(BidirectionalGraph<TVertex, TEdge> wrappedGraph)
    {
        Parent = null;
        Wrapped = wrappedGraph ?? throw new ArgumentNullException(nameof(wrappedGraph));
        Collapsed = false;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ClusteredBidirectionalGraph{TVertex,TEdge}"/> class.
    /// </summary>
    /// <param name="parentGraph">Parent graph.</param>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="parentGraph"/> is <see langword="null"/>.</exception>
    public ClusteredBidirectionalGraph(ClusteredBidirectionalGraph<TVertex, TEdge> parentGraph)
    {
        Parent = parentGraph ?? throw new ArgumentNullException(nameof(parentGraph));
        Wrapped = new BidirectionalGraph<TVertex, TEdge>(parentGraph.AllowParallelEdges);
    }

    public event VertexAction<TVertex> VertexAdded
    {
        add
        {
            Wrapped.VertexAdded += value;
        }

        remove
        {
            Wrapped.VertexAdded -= value;
        }
    }

    public event VertexAction<TVertex> VertexRemoved
    {
        add
        {
            Wrapped.VertexRemoved += value;
        }

        remove
        {
            Wrapped.VertexRemoved -= value;
        }
    }

    public event EdgeAction<TVertex, TEdge> EdgeAdded
    {
        add
        {
            Wrapped.EdgeAdded += value;
        }

        remove
        {
            Wrapped.EdgeAdded -= value;
        }
    }

    public event EdgeAction<TVertex, TEdge> EdgeRemoved
    {
        add
        {
            Wrapped.EdgeRemoved += value;
        }

        remove
        {
            Wrapped.EdgeRemoved -= value;
        }
    }

    /// <summary>
    /// Gets Parent graph.
    /// </summary>
    public ClusteredBidirectionalGraph<TVertex, TEdge>? Parent { get; }

    /// <summary>
    /// Gets or sets the edge capacity.
    /// </summary>
    public int EdgeCapacity
    {
        get => Wrapped.EdgeCapacity;
        set => Wrapped.EdgeCapacity = value;
    }

    /// <summary>
    /// Gets the type of vertices.
    /// </summary>
    public Type VertexType => typeof(TVertex);

    /// <summary>
    /// Gets the type of edges.
    /// </summary>
    public Type EdgeType => typeof(TEdge);

    /// <inheritdoc />
    public bool IsDirected => Wrapped.IsDirected;

    /// <inheritdoc />
    public bool AllowParallelEdges => Wrapped.AllowParallelEdges;

    /// <inheritdoc />
    public bool Collapsed { get; set; }

    /// <inheritdoc />
    public IEnumerable Clusters => _clusters.AsEnumerable();

    /// <inheritdoc />
    public int ClustersCount => _clusters.Count;

    /// <inheritdoc />
    public bool IsVerticesEmpty => Wrapped.IsVerticesEmpty;

    /// <inheritdoc />
    public int VertexCount => Wrapped.VertexCount;

    /// <inheritdoc />
    public virtual IEnumerable<TVertex> Vertices => Wrapped.Vertices;

    /// <inheritdoc />
    public bool IsEdgesEmpty => Wrapped.IsEdgesEmpty;

    /// <inheritdoc />
    public int EdgeCount => Wrapped.EdgeCount;

    /// <inheritdoc />
    public virtual IEnumerable<TEdge> Edges => Wrapped.Edges;

    /// <summary>
    /// Gets Wrapped graph.
    /// </summary>
    protected BidirectionalGraph<TVertex, TEdge> Wrapped { get; }

    /// <inheritdoc />
    public bool ContainsEdge(TEdge edge)
    {
        return Wrapped.ContainsEdge(edge);
    }

    /// <summary>
    /// Adds a new cluster.
    /// </summary>
    /// <returns>The added cluster.</returns>
    public ClusteredBidirectionalGraph<TVertex, TEdge> AddCluster()
    {
        var cluster = new ClusteredBidirectionalGraph<TVertex, TEdge>(this);
        _clusters.Add(cluster);
        return cluster;
    }

    /// <inheritdoc />
    IClusteredGraph IClusteredGraph.AddCluster()
    {
        return AddCluster();
    }

    /// <inheritdoc />
    public void RemoveCluster(IClusteredGraph graph)
    {
        if (graph is null)
        {
            throw new ArgumentNullException(nameof(graph));
        }

        _clusters.Remove(graph);
    }

    /// <inheritdoc />
    public bool ContainsVertex(TVertex vertex)
    {
        return Wrapped.ContainsVertex(vertex);
    }

    /// <inheritdoc />
    public bool ContainsEdge(TVertex source, TVertex target)
    {
        return Wrapped.ContainsEdge(source, target);
    }

    /// <inheritdoc />
    public bool TryGetEdge(TVertex source, TVertex target, out TEdge edge)
    {
        return Wrapped.TryGetEdge(source, target, out edge);
    }

    /// <inheritdoc />
    public virtual bool TryGetEdges(TVertex source, TVertex target, out IEnumerable<TEdge> edges)
    {
        return Wrapped.TryGetEdges(source, target, out edges);
    }

    /// <inheritdoc />
    public bool IsOutEdgesEmpty(TVertex vertex)
    {
        return Wrapped.IsOutEdgesEmpty(vertex);
    }

    /// <inheritdoc />
    public int OutDegree(TVertex vertex)
    {
        return Wrapped.OutDegree(vertex);
    }

    /// <inheritdoc />
    public virtual IEnumerable<TEdge> OutEdges(TVertex vertex)
    {
        return Wrapped.OutEdges(vertex);
    }

    /// <inheritdoc />
    public virtual bool TryGetOutEdges(TVertex vertex, out IEnumerable<TEdge> edges)
    {
        return Wrapped.TryGetOutEdges(vertex, out edges);
    }

    /// <inheritdoc />
    public TEdge OutEdge(TVertex vertex, int index)
    {
        return Wrapped.OutEdge(vertex, index);
    }

    /// <summary>
    /// Adds a vertex to this graph.
    /// </summary>
    /// <param name="vertex">Vertex to add.</param>
    /// <returns>True if the vertex was added, false otherwise.</returns>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="vertex"/> is <see langword="null"/>.</exception>
    public virtual bool AddVertex(TVertex vertex)
    {
        if (vertex == null)
        {
            throw new ArgumentNullException(nameof(vertex));
        }

        if (!(Parent is null || Parent.ContainsVertex(vertex)))
        {
            Parent.AddVertex(vertex);
            return Wrapped.AddVertex(vertex);
        }

        return Wrapped.AddVertex(vertex);
    }

    /// <summary>
    /// Adds given vertices to this graph.
    /// </summary>
    /// <param name="vertices">Vertices to add.</param>
    /// <returns>The number of vertex added.</returns>
    /// <exception cref="T:System.ArgumentNullException">
    /// <paramref name="vertices"/> is <see langword="null"/> or at least one of them is <see langword="null"/>.
    /// </exception>
    public virtual int AddVertexRange([NotNull] IEnumerable<TVertex> vertices)
    {
        if (vertices is null)
        {
            throw new ArgumentNullException(nameof(vertices));
        }

        TVertex[] verticesArray = vertices.ToArray();
        if (verticesArray.Any(v => v == null))
        {
            throw new ArgumentNullException(nameof(vertices), "At least one vertex is null.");
        }

        return verticesArray.Count(AddVertex);
    }

    /// <summary>
    /// Removes the given vertex from this graph.
    /// </summary>
    /// <param name="vertex">Vertex to remove.</param>
    /// <returns>True if the vertex was removed, false otherwise.</returns>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="vertex"/> is <see langword="null"/>.</exception>
    public virtual bool RemoveVertex(TVertex vertex)
    {
        if (vertex == null)
        {
            throw new ArgumentNullException(nameof(vertex));
        }

        if (!Wrapped.ContainsVertex(vertex))
        {
            return false;
        }

        RemoveVertexInternal(vertex);

        return true;
    }

    /// <summary>
    /// Removes all vertices matching the given <paramref name="predicate"/>.
    /// </summary>
    /// <param name="predicate">Predicate to check on each vertex.</param>
    /// <returns>The number of vertex removed.</returns>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="predicate"/> is <see langword="null"/>.</exception>
    public int RemoveVertexIf([NotNull] VertexPredicate<TVertex> predicate)
    {
        if (predicate is null)
        {
            throw new ArgumentNullException(nameof(predicate));
        }

        var verticesToRemove = new VertexList<TVertex>();
        verticesToRemove.AddRange(Vertices.Where(vertex => predicate(vertex)));

        foreach (TVertex vertex in verticesToRemove)
        {
            RemoveVertexInternal(vertex);
        }

        return verticesToRemove.Count;
    }

    /// <summary>
    /// Adds <paramref name="edge"/> and its vertices to this graph.
    /// </summary>
    /// <param name="edge">The edge to add.</param>
    /// <returns>True if the edge was added, false otherwise.</returns>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="edge"/> is <see langword="null"/>.</exception>
    public virtual bool AddVerticesAndEdge(TEdge edge)
    {
        if (edge == null)
        {
            throw new ArgumentNullException(nameof(edge));
        }

        AddVertex(edge.Source);
        AddVertex(edge.Target);
        return AddEdge(edge);
    }

    /// <summary>
    /// Adds a set of edges (and it's vertices if necessary).
    /// </summary>
    /// <param name="edges">Edges to add.</param>
    /// <returns>The number of edges added.</returns>
    /// <exception cref="T:System.ArgumentNullException">
    /// <paramref name="edges"/> is <see langword="null"/> or at least one of them is <see langword="null"/>.
    /// </exception>
    public int AddVerticesAndEdgeRange([NotNull] IEnumerable<TEdge> edges)
    {
        if (edges is null)
        {
            throw new ArgumentNullException(nameof(edges));
        }

        TEdge[] edgesArray = edges.ToArray();
        if (edgesArray.Any(e => e == null))
        {
            throw new ArgumentNullException(nameof(edges), "At least one edge is null.");
        }

        return edgesArray.Count(AddVerticesAndEdge);
    }

    /// <summary>
    /// Adds the <paramref name="edge"/> to this graph.
    /// </summary>
    /// <param name="edge">An edge.</param>
    /// <returns>True if the edge was added, false otherwise.</returns>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="edge"/> is <see langword="null"/>.</exception>
    public virtual bool AddEdge(TEdge edge)
    {
        if (edge == null)
        {
            throw new ArgumentNullException(nameof(edge));
        }

        if (Parent != null && !Parent.ContainsEdge(edge))
        {
            Parent.AddEdge(edge);
        }

        return Wrapped.AddEdge(edge);
    }

    /// <summary>
    /// Adds a set of edges to this graph.
    /// </summary>
    /// <param name="edges">Edges to add.</param>
    /// <returns>The number of edges successfully added to this graph.</returns>
    /// <exception cref="T:System.ArgumentNullException">
    /// <paramref name="edges"/> is <see langword="null"/> or at least one of them is <see langword="null"/>.
    /// </exception>
    public int AddEdgeRange([NotNull] IEnumerable<TEdge> edges)
    {
        if (edges is null)
        {
            throw new ArgumentNullException(nameof(edges));
        }

        TEdge[] edgesArray = edges.ToArray();
        if (edgesArray.Any(e => e == null))
        {
            throw new ArgumentNullException(nameof(edges), "At least one edge is null.");
        }

        return edgesArray.Count(AddEdge);
    }

    /// <summary>
    /// Removes the <paramref name="edge"/> from this graph.
    /// </summary>
    /// <param name="edge">Edge to remove.</param>
    /// <returns>True if the <paramref name="edge"/> was successfully removed, false otherwise.</returns>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="edge"/> is <see langword="null"/>.</exception>
    public virtual bool RemoveEdge(TEdge edge)
    {
        if (edge == null)
        {
            throw new ArgumentNullException(nameof(edge));
        }

        if (!Wrapped.ContainsEdge(edge))
        {
            return false;
        }

        RemoveEdgeInternal(edge);

        return true;
    }

    /// <summary>
    /// Removes all edges that match the given <paramref name="predicate"/>.
    /// </summary>
    /// <param name="predicate">Predicate to check if an edge should be removed.</param>
    /// <returns>The number of edges removed.</returns>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="predicate"/> is <see langword="null"/>.</exception>
    public int RemoveEdgeIf([NotNull] EdgePredicate<TVertex, TEdge> predicate)
    {
        if (predicate is null)
        {
            throw new ArgumentNullException(nameof(predicate));
        }

        var edgesToRemove = new EdgeList<TVertex, TEdge>();
        edgesToRemove.AddRange(Edges.Where(edge => predicate(edge)));

        foreach (TEdge edge in edgesToRemove)
        {
            RemoveEdgeInternal(edge);
        }

        return edgesToRemove.Count;
    }

    /// <summary>
    /// Removes all out-edges of the <paramref name="vertex"/>
    /// where the <paramref name="predicate"/> is evaluated to true.
    /// </summary>
    /// <param name="vertex">The vertex.</param>
    /// <param name="predicate">Predicate to remove edges.</param>
    /// <returns>The number of removed edges.</returns>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="vertex"/> is <see langword="null"/>.</exception>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="predicate"/> is <see langword="null"/>.</exception>
    public int RemoveOutEdgeIf(TVertex vertex, [NotNull] EdgePredicate<TVertex, TEdge> predicate)
    {
        if (vertex == null)
        {
            throw new ArgumentNullException(nameof(vertex));
        }

        if (predicate is null)
        {
            throw new ArgumentNullException(nameof(predicate));
        }

        int edgeToRemoveCount = Wrapped.RemoveOutEdgeIf(vertex, predicate);
        Parent?.RemoveOutEdgeIf(vertex, predicate);

        return edgeToRemoveCount;
    }

    /// <summary>
    /// Clears the out-edges of the given <paramref name="vertex"/>.
    /// </summary>
    /// <param name="vertex">The vertex.</param>
    /// <exception cref="T:System.ArgumentNullException"><paramref name="vertex"/> is <see langword="null"/>.</exception>
    public void ClearOutEdges(TVertex vertex)
    {
        if (vertex == null)
        {
            throw new ArgumentNullException(nameof(vertex));
        }

        Wrapped.ClearOutEdges(vertex);
    }

    /// <summary>
    /// Clears the vertex and edges.
    /// </summary>
    public void Clear()
    {
        Wrapped.Clear();
        _clusters.Clear();
    }

    /// <inheritdoc/>
    public int RemoveInEdgeIf(TVertex vertex, EdgePredicate<TVertex, TEdge> predicate) => Wrapped.RemoveInEdgeIf(vertex, predicate);

    /// <inheritdoc/>
    public void ClearInEdges(TVertex vertex) => Wrapped.ClearInEdges(vertex);

    /// <inheritdoc/>
    public void ClearEdges(TVertex vertex) => Wrapped.ClearEdges(vertex);

    /// <inheritdoc/>
    public void TrimEdgeExcess() => Wrapped.TrimEdgeExcess();

    /// <inheritdoc/>
    public bool IsInEdgesEmpty(TVertex vertex) => Wrapped.IsInEdgesEmpty(vertex);

    /// <inheritdoc/>
    public int InDegree(TVertex vertex) => Wrapped.InDegree(vertex);

    /// <inheritdoc/>
    public IEnumerable<TEdge> InEdges(TVertex vertex) => Wrapped.InEdges(vertex);

    /// <inheritdoc/>
    public bool TryGetInEdges(TVertex vertex, out IEnumerable<TEdge> edges) => Wrapped.TryGetInEdges(vertex, out edges);

    /// <inheritdoc/>
    public TEdge InEdge(TVertex vertex, int index) => Wrapped.InEdge(vertex, index);

    /// <inheritdoc/>
    public int Degree(TVertex vertex) => Wrapped.Degree(vertex);

    private void RemoveChildEdge(TEdge edge)
    {
        Debug.Assert(edge != null, "edge can't be null");

        foreach (ClusteredBidirectionalGraph<TVertex, TEdge> cluster in Clusters)
        {
            if (cluster.ContainsEdge(edge))
            {
                cluster.Wrapped.RemoveEdge(edge);
                cluster.RemoveChildEdge(edge);
            }
        }
    }

#if SUPPORTS_AGGRESSIVE_INLINING
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
    private void RemoveEdgeInternal(TEdge edge)
    {
        Debug.Assert(edge != null, "edge can't be null");

        RemoveChildEdge(edge);
        Wrapped.RemoveEdge(edge);
        Parent?.RemoveEdge(edge);
    }

    /// <summary>
    /// Removes the given vertex from clusters.
    /// </summary>
    /// <param name="vertex">Vertex to remove.</param>
    private void RemoveChildVertex(TVertex vertex)
    {
        Debug.Assert(vertex != null, "vertex can't be null");

        foreach (ClusteredBidirectionalGraph<TVertex, TEdge> cluster in Clusters)
        {
            if (cluster.ContainsVertex(vertex))
            {
                cluster.Wrapped.RemoveVertex(vertex);
                cluster.RemoveChildVertex(vertex);
            }
        }
    }

#if SUPPORTS_AGGRESSIVE_INLINING
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#endif
    private void RemoveVertexInternal(TVertex vertex)
    {
        Debug.Assert(vertex != null, "vertex can't be null");

        RemoveChildVertex(vertex);
        Wrapped.RemoveVertex(vertex);
        Parent?.RemoveVertex(vertex);
    }
}
