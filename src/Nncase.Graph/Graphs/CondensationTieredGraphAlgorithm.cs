// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Algorithms.Search;
using QuikGraph.Collections;

namespace Nncase.Graphs;

public sealed class CondensationTieredGraphAlgorithm<TGraph, TVertex, TEdge> : AlgorithmBase<TGraph>
    where TEdge : IEdge<TVertex>
    where TVertex : notnull
    where TGraph : TieredAdjacencyGraph<TVertex, TEdge>
{
    private readonly Dictionary<TVertex, TGraph> _vertexMap = new();

    public CondensationTieredGraphAlgorithm(TGraph visitedGraph)
     : base(visitedGraph)
    {
        CondensedGraph = new(false);
    }

    public AdjacencyGraph<TGraph, Edge<TGraph>> CondensedGraph { get; private set; }

    protected override void InternalCompute()
    {
        if (VisitedGraph.VertexCount == 0)
        {
            return;
        }

        foreach (var primGraph in VisitedGraph.Clusters.OfType<TGraph>())
        {
            foreach (var vertex in primGraph.Vertices)
            {
                if (!_vertexMap.TryGetValue(vertex, out _))
                {
                    _vertexMap.Add(vertex, primGraph);
                }
            }

            CondensedGraph.AddVertex(primGraph);
        }

        var dfs = new DepthFirstSearchAlgorithm<TVertex, TEdge>(this, VisitedGraph, new Dictionary<TVertex, GraphColor>(VisitedGraph.VertexCount));
        dfs.TreeEdge += TreeEdge;
        dfs.Compute();
    }

    private void TreeEdge(TEdge edge)
    {
        var sourceGraph = _vertexMap[edge.Source];
        var targetGraph = _vertexMap[edge.Target];
        if (!ReferenceEquals(sourceGraph, targetGraph))
        {
            CondensedGraph.AddEdge(new(sourceGraph, targetGraph));
        }
    }
}
