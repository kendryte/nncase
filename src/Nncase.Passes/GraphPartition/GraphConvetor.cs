// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.IR.Affine;
using QuikGraph;
using QuikGraph.Graphviz;

namespace Nncase.Passes.GraphPartition;

public sealed class GraphContext
{
    public Graph Graph { get; set; } = new();

    public Graph GraphSummary { get; set; } = new();

    public SortedDictionary<int, Subgraph> SubgraphMap { get; set; } = new();

    public Dictionary<Vertex, int> OriginalVertexSubgraphMap { get; set; } = new();

    public Dictionary<Vertex, int> SummaryVertexSubgraphMap { get; set; } = new();

    public Dictionary<int, Dictionary<Expr, Var>> VarMap { get; set; } = new();

    public Dictionary<int, Dictionary<Expr, int>> OutputMap { get; set; } = new();

    public void MergeSubgraphMap()
    {
        OriginalVertexSubgraphMap = Graph.Vertices.Select((v, i) => new KeyValuePair<Vertex, int>(v, i)).ToDictionary(kv => kv.Key, kv => kv.Value);

        // Create subgraph structs
        var dfsAssignEdge = new QuikGraph.Algorithms.Search.EdgeDepthFirstSearchAlgorithm<Vertex, Edge>(Graph);
        dfsAssignEdge.TreeEdge += (edge) =>
        {
            var u = edge.Source;
            var v = edge.Target;

            var u_sub_idx = OriginalVertexSubgraphMap[u];
            var v_sub_idx = OriginalVertexSubgraphMap[v];

            if (u_sub_idx == v_sub_idx)
            {
                SubgraphMap[u_sub_idx].InteriorEdges.Add(edge);
            }
            else
            {
                SubgraphMap[u_sub_idx].OutputEdges.Add(edge);
                SubgraphMap[v_sub_idx].InputEdges.Add(edge);
            }
        };
        dfsAssignEdge.Compute();

        var dfsVisitEdge = new QuikGraph.Algorithms.Search.EdgeDepthFirstSearchAlgorithm<Vertex, Edge>(Graph);
        dfsVisitEdge.InitializeEdge += (edge) =>
        {
            var u = edge.Source;
            var v = edge.Target;

            if (u.CompatType != v.CompatType)
            {
                return;
            }

            if (OriginalVertexSubgraphMap[u] == OriginalVertexSubgraphMap[v])
            {
                return;
            }

            var tmpSubgraphMap = new SortedDictionary<int, Subgraph>(SubgraphMap.Comparer);
            var tmpvertexSubgraphMap = new Dictionary<Vertex, int>(OriginalVertexSubgraphMap, OriginalVertexSubgraphMap.Comparer);
            foreach (var kvp in SubgraphMap)
            {
                tmpSubgraphMap[kvp.Key] = new Subgraph(kvp.Value.Index, new List<Vertex>(kvp.Value.Nodes), new List<Edge>(kvp.Value.InputEdges), new List<Edge>(kvp.Value.OutputEdges), new List<Edge>(kvp.Value.InteriorEdges));
            }

            // var vExclusiveInputs = v_subgraph.InputEdges.Where(x => !u_subgraph.OutputEdges.Contains(x));
            // if (SubgraphMap.Values.Any(x => vExclusiveInputs.Any(y => x.OutputEdges.Contains(y) && x.InputEdges.Any(z => u_subgraph.OutputEdges.Contains(z)))))
            // {
            //     return;
            // }
            var u_subgraph = tmpSubgraphMap[OriginalVertexSubgraphMap[u]];
            var v_subgraph = tmpSubgraphMap[OriginalVertexSubgraphMap[v]];

            MergeTwoSubgraphs(v_subgraph, u_subgraph, tmpSubgraphMap, tmpvertexSubgraphMap);

            if (!HasCycles(tmpSubgraphMap, tmpvertexSubgraphMap))
            {
                SubgraphMap = new SortedDictionary<int, Subgraph>(tmpSubgraphMap, tmpSubgraphMap.Comparer);
                OriginalVertexSubgraphMap = tmpvertexSubgraphMap;
            }
        };
        dfsVisitEdge.Compute();
    }

    public void SummarizeGraph()
    {
        MergeSubgraphMap();

        GraphSummary = new();
        Dictionary<int, Vertex> indexMap = new();
        VarMap = SubgraphMap.ToDictionary(x => x.Key, _ => new Dictionary<Expr, Var>(ReferenceEqualityComparer.Instance));
        OutputMap = SubgraphMap.ToDictionary(x => x.Key, _ => new Dictionary<Expr, int>(ReferenceEqualityComparer.Instance));

        // int count = 0;
        foreach (var subgraph in SubgraphMap)
        {
            var sg = new Graph();
            subgraph.Value.Nodes.ForEach(n => sg.AddVertex(n));
            subgraph.Value.InteriorEdges.ForEach(e => sg.AddEdge(e));

            // sg.DumpDot(Diagnostics.DumpScope.Current.Directory + $"subgraph_{subgraph.Key}_{count++}.dot");
            var dfsVisitor = new QuikGraph.Algorithms.TopologicalSort.SourceFirstTopologicalSortAlgorithm<Vertex, Edge>(sg);
            dfsVisitor.Compute();
            for (var vi = 0; vi < dfsVisitor.SortedVertices.Length; vi++)
            {
                var vertex = dfsVisitor.SortedVertices[vi];
                if (vertex.Expr is Var v)
                {
                    if (!VarMap[subgraph.Key].ContainsKey(v))
                    {
                        VarMap[subgraph.Key].Add(v, new Var(v.CheckedType));
                    }
                }
                else if (subgraph.Value.InputEdges.Any(e => e.Target == vertex))
                {
                    foreach (var input in subgraph.Value.InputEdges.Where(e => e.Target == vertex).Select(e => e.Source.Expr))
                    {
                        if (input is not Const && !VarMap[subgraph.Key].ContainsKey(input))
                        {
                            if (input.CheckedType is DistributedType d)
                            {
                                VarMap[subgraph.Key].Add(input, new Var(d.TensorType));
                            }
                            else
                            {
                                VarMap[subgraph.Key].Add(input, new Var(input.CheckedType));
                            }
                        }
                    }
                }
            }

            var u = new Vertex(None.Default, Compat.UNKNOWN);
            var outVertices = sg.Vertices.Count() == 1 ? sg.Vertices : subgraph.Value.OutputEdges.Select(e => e.Source).Distinct();
            if (!outVertices.Any())
            {
                outVertices = sg.Edges.Where(e => !sg.OutEdges(e.Target).Any()).Select(e => e.Target).Distinct().ToList();
            }

            u.CompatType = sg.Vertices.First().CompatType;

            if (outVertices.Count() == 1)
            {
                u.Expr = outVertices.First().Expr;
                if (u.CompatType == Compat.COMPATIBLE)
                {
                    OutputMap[subgraph.Key].Add(u.Expr, -1);
                }
            }
            else
            {
                u.Expr = new IR.Tuple(outVertices.Select(x => x.Expr).ToArray());
                if (u.CompatType == Compat.COMPATIBLE)
                {
                    Enumerable.Range(0, outVertices.Count()).ToList().ForEach(i => OutputMap[subgraph.Key].Add(outVertices.ToList()[i].Expr, i));
                }
            }

            SummaryVertexSubgraphMap.Add(u, subgraph.Key);

            indexMap.Add(subgraph.Key, u);
            GraphSummary.AddVertex(u);
        }

        Dictionary<Edge, Edge> edgeMap = new();
        foreach (var subgraph in SubgraphMap)
        {
            foreach (var edge in subgraph.Value.OutputEdges)
            {
                var u = indexMap[OriginalVertexSubgraphMap[edge.Source]];
                var v = indexMap[OriginalVertexSubgraphMap[edge.Target]];

                var newEdge = new Edge(edge.EdgeType, u, v);
                GraphSummary.AddEdge(newEdge);
                if (edgeMap.ContainsKey(newEdge))
                {
                    // System.Console.WriteLine("[ERROR] " + edge + " already mapped!");
                }
                else
                {
                    edgeMap.Add(newEdge, edge);
                }
            }
        }
    }

    private void MergeTwoSubgraphs(Subgraph target, Subgraph source, SortedDictionary<int, Subgraph> subgraphMap, Dictionary<Vertex, int> vertexSubgraphMap)
    {
        source.Nodes.ForEach(x => vertexSubgraphMap[x] = target.Index);

        target.Nodes.AddRange(source.Nodes);

        var mergedEdges = source.OutputEdges.Where(s => target.InputEdges.Contains(s)).ToList();
        target.InteriorEdges.AddRange(mergedEdges);
        target.InteriorEdges.AddRange(source.InteriorEdges);

        mergedEdges.ForEach(x => target.InputEdges.Remove(x));
        target.InputEdges.AddRange(source.InputEdges);

        source.OutputEdges.ForEach(x =>
        {
            if (!mergedEdges.Contains(x))
            {
                target.OutputEdges.Add(x);
            }
        });

        subgraphMap.Remove(source.Index);
    }

    private bool HasCycles(SortedDictionary<int, Subgraph> subgraphMap, Dictionary<Vertex, int> vertexSubgraphMap)
    {
        var graphSummary = new Graph();
        Dictionary<int, Vertex> indexMap = new();
        foreach (var subgraph in subgraphMap)
        {
            var u = new Vertex(new Var(), subgraph.Value.Nodes[0].CompatType);
            indexMap.Add(subgraph.Key, u);
            graphSummary.AddVertex(u);
        }

        foreach (var subgraph in subgraphMap)
        {
            foreach (var edge in subgraph.Value.OutputEdges)
            {
                var u = indexMap[vertexSubgraphMap[edge.Source]];
                var v = indexMap[vertexSubgraphMap[edge.Target]];

                var newEdge = new Edge(edge.EdgeType, u, v);
                graphSummary.AddEdge(newEdge);
            }
        }

        List<Edge> cycles = new();
        var dfs = new QuikGraph.Algorithms.Search.EdgeDepthFirstSearchAlgorithm<Vertex, Edge>(graphSummary);
        dfs.BackEdge += (edge) =>
        {
            var u = edge.Source;
            var v = edge.Target;
            cycles.Add(edge);
        };
        dfs.Compute();

        return cycles.Count > 0;
    }
}

public sealed class GraphConvertor : ExprVisitor<Unit, Unit, GraphContext>
{
    private int _nodeCount;

    public GraphConvertor(Func<Expr, bool> predicate)
    {
        Predicate = predicate;
    }

    public Func<Expr, bool> Predicate { get; }

    protected override Unit VisitGrid(Grid expr, GraphContext context)
    {
        foreach (var operand in expr.Reads)
        {
            Visit(operand, context);
        }

        return VisitLeafGrid(expr, context);
    }

    protected override Unit VisitLeafGrid(Grid expr, GraphContext context)
    {
        Vertex target;
        if (Predicate(expr))
        {
            target = new Vertex(expr, Compat.COMPATIBLE);
        }
        else
        {
            target = new Vertex(expr, Compat.INCOMPATIBLE);
        }

        context.Graph.AddVertex(target);
        context.SubgraphMap.Add(_nodeCount, new Subgraph(_nodeCount, new() { target }, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        _nodeCount++;
        foreach (var operand in expr.Reads)
        {
            if (context.Graph.Vertices.Any(v => ReferenceEquals(v.Expr, operand)))
            {
                var source = context.Graph.Vertices.First(v => ReferenceEquals(v.Expr, operand));
                switch (source.CompatType, target.CompatType)
                {
                    case (Compat.COMPATIBLE, Compat.INCOMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.C2I, source, target));
                        break;
                    case (Compat.INCOMPATIBLE, Compat.COMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.I2C, source, target));
                        break;
                    case (Compat.INCOMPATIBLE, Compat.INCOMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.I2I, source, target));
                        break;
                    default:
                        context.Graph.AddEdge(new Edge(EdgeTypes.C2C, source, target));
                        break;
                }
            }
        }

        return default;
    }

    protected override Unit VisitLeafVar(Var expr, GraphContext context)
    {
        Vertex target;

        target = new Vertex(expr, Compat.INCOMPATIBLE);

        context.Graph.AddVertex(target);
        context.SubgraphMap.Add(_nodeCount, new Subgraph(_nodeCount, new() { target }, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        _nodeCount++;

        return default;
    }

    protected override Unit VisitLeafConst(Const expr, GraphContext context)
    {
        Vertex target;
        target = new Vertex(expr, expr.CheckedType is DistributedType ? Compat.COMPATIBLE : Compat.INCOMPATIBLE);

        context.Graph.AddVertex(target);
        context.SubgraphMap.Add(_nodeCount, new Subgraph(_nodeCount, new() { target }, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        _nodeCount++;

        return default;
    }

    protected override Unit VisitLeafNone(None expr, GraphContext context)
    {
        Vertex target;
        target = new Vertex(expr, Compat.INCOMPATIBLE);

        context.Graph.AddVertex(target);
        context.SubgraphMap.Add(_nodeCount, new Subgraph(_nodeCount, new() { target }, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        _nodeCount++;

        return default;
    }

    protected override Unit VisitLeafCall(Call expr, GraphContext context)
    {
        Vertex target;
        if (Predicate(expr))
        {
            target = new Vertex(expr, Compat.COMPATIBLE);
        }
        else
        {
            target = new Vertex(expr, Compat.INCOMPATIBLE);
        }

        context.Graph.AddVertex(target);
        context.SubgraphMap.Add(_nodeCount, new Subgraph(_nodeCount, new() { target }, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        _nodeCount++;
        foreach (var operand in expr.Arguments)
        {
            if (context.Graph.Vertices.Any(v => ReferenceEquals(v.Expr, operand)))
            {
                var source = context.Graph.Vertices.First(v => ReferenceEquals(v.Expr, operand));
                switch (source.CompatType, target.CompatType)
                {
                    case (Compat.COMPATIBLE, Compat.INCOMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.C2I, source, target));
                        break;
                    case (Compat.INCOMPATIBLE, Compat.COMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.I2C, source, target));
                        break;
                    case (Compat.INCOMPATIBLE, Compat.INCOMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.I2I, source, target));
                        break;
                    default:
                        context.Graph.AddEdge(new Edge(EdgeTypes.C2C, source, target));
                        break;
                }
            }
        }

        return default;
    }

    protected override Unit VisitLeafTuple(IR.Tuple expr, GraphContext context)
    {
        Vertex target;
        var compatType = context.Graph.Vertices.First(v => ReferenceEquals(v.Expr, expr.Fields[0])).CompatType;
        if (!Predicate(expr))
        {
            compatType = Compat.INCOMPATIBLE;
        }

        target = new Vertex(expr, compatType);

        context.Graph.AddVertex(target);
        context.SubgraphMap.Add(_nodeCount, new Subgraph(_nodeCount, new() { target }, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        _nodeCount++;
        foreach (var field in expr.Fields)
        {
            if (context.Graph.Vertices.Any(v => ReferenceEquals(v.Expr, field)))
            {
                var source = context.Graph.Vertices.First(v => ReferenceEquals(v.Expr, field));
                switch (source.CompatType, target.CompatType)
                {
                    case (Compat.COMPATIBLE, Compat.INCOMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.C2I, source, target));
                        break;
                    case (Compat.INCOMPATIBLE, Compat.COMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.I2C, source, target));
                        break;
                    case (Compat.INCOMPATIBLE, Compat.INCOMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.I2I, source, target));
                        break;
                    default:
                        context.Graph.AddEdge(new Edge(EdgeTypes.C2C, source, target));
                        break;
                }
            }
        }

        return default;
    }

    protected override Unit DefaultVisitLeaf(Expr expr, GraphContext context)
    {
        return default;
    }
}
