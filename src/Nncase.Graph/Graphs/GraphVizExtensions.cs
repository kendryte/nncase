// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.Graphs;
using QuikGraph;
using QuikGraph.Graphviz;

namespace Nncase.Graphs;

public static class GraphVizExtensions
{
    public static void Dump<TVertex, TEdge>(this IEdgeListGraph<TVertex, TEdge> graph, string name, Action<GraphvizAlgorithm<TVertex, TEdge>>? initAlgorithm = null)
        where TEdge : IEdge<TVertex>
    {
        Action<GraphvizAlgorithm<TVertex, TEdge>> empty = (_) => { };
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"{name}.dot"))
        {
            using (var writer = new StreamWriter(stream))
            {
                writer.Write(graph.ToGraphviz(initAlgorithm ?? empty));
            }
        }
    }
}
