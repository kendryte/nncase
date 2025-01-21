// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Collections.Generic;
using System.Linq;
using QuikGraph;

namespace Nncase.Graphs;

public static class GraphExtensions
{
    public static IEnumerable<TEdge> InEdges<TVertex, TEdge>(this IBidirectionalGraph<TVertex, TEdge> subGraph, IBidirectionalGraph<TVertex, TEdge> parentGraph)
         where TEdge : IEdge<TVertex> => subGraph.Vertices.Select(v => parentGraph.InEdges(v).Except(subGraph.InEdges(v))).SelectMany(e => e);

    public static IEnumerable<TEdge> OutEdges<TVertex, TEdge>(this IBidirectionalGraph<TVertex, TEdge> subGraph, IBidirectionalGraph<TVertex, TEdge> parentGraph)
         where TEdge : IEdge<TVertex> => subGraph.Vertices.Select(v => parentGraph.OutEdges(v)).SelectMany(e => e).Where(e => !subGraph.ContainsVertex(e.Target));

    public static IEnumerable<TVertex> InVertices<TVertex, TEdge>(this IBidirectionalGraph<TVertex, TEdge> graph)
         where TEdge : IEdge<TVertex>
        => graph.Vertices.Where(v => graph.InDegree(v) == 0);

    public static IEnumerable<TVertex> OutVertices<TVertex, TEdge>(this IBidirectionalGraph<TVertex, TEdge> graph)
         where TEdge : IEdge<TVertex>
        => graph.Vertices.Where(v => graph.OutDegree(v) == 0);

    public static IEnumerable<TVertex> OutVertices<TVertex, TEdge>(this IBidirectionalGraph<TVertex, TEdge> subGraph, IBidirectionalGraph<TVertex, TEdge> parentGraph)
         where TEdge : IEdge<TVertex>
    {
        var outEdges = OutEdges(subGraph, parentGraph).ToArray();
        if (outEdges.Length == 0)
        {
            return OutVertices(subGraph);
        }

        return outEdges.DistinctBy(e => e.Source).Select(e => e.Source);
    }
}
