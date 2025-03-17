// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.Tests.TestFixture;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Graphviz;
using Xunit;

namespace Nncase.Tests.Rules.ShapeBucket;

[AutoSetupTestMethod(InitSession = true)]
public sealed class CondenstGraphTest : TransformTestBase
{
    [Fact]
    public void TestCondensateWeakly()
    {
        var graph = new AdjacencyGraph<int, Edge<int>>();
        graph.AddVertexRange(new[] { 0, 1, 2, 3, 4, 5 });
        graph.AddEdge(new(0, 1));
        graph.AddEdge(new(1, 2));
        graph.AddEdge(new(0, 3));
        graph.AddEdge(new(3, 2));
        graph.AddEdge(new(4, 5));
        using (var stream = Diagnostics.DumpScope.Current.OpenFile("pre.dot"))
        {
            using var writer = new StreamWriter(stream);
            writer.Write(graph.ToGraphviz());
        }

        var result = graph.CondensateWeaklyConnected<int, Edge<int>, AdjacencyGraph<int, Edge<int>>>();
        using (var stream = Diagnostics.DumpScope.Current.OpenFile("post.dot"))
        {
            using var writer = new StreamWriter(stream);
            writer.Write(result.ToGraphviz(algo =>
            {
                algo.FormatVertex += (_, args) => args.VertexFormat.Label = args.Vertex.ToString();
            }));
        }
    }
}
