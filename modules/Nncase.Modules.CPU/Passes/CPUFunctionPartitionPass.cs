// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.Diagnostics;
using Nncase.IR;
using QuikGraph;
using QuikGraph.Graphviz;

namespace Nncase.Passes;

public enum Compat
{
    UNKNOWN,
    COMPATIBLE,
    INCOMPATIBLE,
    BOUNDARY,
    DATA,
}

public enum EdgeTypes
{
    UNKNOWN,
    C2C,
    I2I,
    C2I,
    I2C,
}

public sealed record Vertex
{
    public Vertex(Expr expr, Compat compatType, int subgraphIndex)
    {
        Expr = expr;
        CompatType = compatType;
        SubgraphIndex = subgraphIndex;
    }

    public Expr Expr { get; set; }

    public Compat CompatType { get; set; }

    public int SubgraphIndex { get; set; }
}

public sealed record Edge : QuikGraph.IEdge<Vertex>
{
    public Edge(EdgeTypes edgeType, int index, Vertex source, Vertex target)
    {
        EdgeType = edgeType;
        Index = index;
        Source = source;
        Target = target;
    }

    public EdgeTypes EdgeType { get; set; }

    public int Index { get; set; }

    public Vertex Source { get; }

    public Vertex Target { get; }
}

public sealed class Graph : AdjacencyGraph<Vertex, Edge>
{
}

public sealed record Subgraph(List<Vertex> Nodes, List<Edge> InputEdges, List<Edge> OutputEdges, List<Edge> InteriorEdges);

public sealed class CPUFunctionPartitionPass : ModulePass
{
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext context)
    {
        var funcs = module.Functions.Count;
        for (int i = 0; i < funcs; i++)
        {
            if (module.Functions[i] is Function function)
            {
                Function pre = function;

                // Function post;
                var ctx = new GraphContext();
                var convertor = new GraphConvertor();
                convertor.Visit(pre.Body, ctx);

                using (var writer = new StreamWriter(DumpScope.Current.Directory + $"graph_{i}.dot"))
                {
                    var a = ctx.Graph.ToGraphviz<Vertex, Edge>();
                    writer.Write(a);
                }

                // module.Replace(i, post);
            }
        }

        return Task.FromResult(module);
    }
}

internal sealed class GraphContext
{
    public Graph Graph { get; set; } = new();

    public Graph GraphSummary { get; set; } = new();

    public Dictionary<int, List<Vertex>> SubgraphNodes { get; set; } = new();

    public SortedDictionary<int, Subgraph> SubgraphMap { get; set; } = new();

    public List<Subgraph> Subgraphs { get; set; } = new();

    public void MergeSubgraphMap()
    {
        var dfsVisitEdge = new QuikGraph.Algorithms.Search.EdgeDepthFirstSearchAlgorithm<Vertex, Edge>(Graph);
        dfsVisitEdge.TreeEdge += (edge) =>
        {
            var u = edge.Source;
            var v = edge.Target;

            if (u.CompatType == v.CompatType)
            {
                return;
            }

            if (u.SubgraphIndex == v.SubgraphIndex)
            {
                return;
            }

            var u_subgraph = SubgraphNodes[u.SubgraphIndex];
            var v_subgraph = SubgraphNodes[v.SubgraphIndex];
            foreach (var v_s in v_subgraph)
            {
                u_subgraph.Add(v_s);
                v_s.SubgraphIndex = u.SubgraphIndex;
            }

            v_subgraph.Clear();
        };
        dfsVisitEdge.Compute();

        // remove empty subgraphs
        var subgraph_rm_list = SubgraphNodes.Where(x => x.Value.Count == 0).Select(x => x.Key).ToList();
        subgraph_rm_list.ForEach(x => SubgraphNodes.Remove(x));

        // Create subgraph structs
        foreach (var subgraph in SubgraphNodes)
        {
            SubgraphMap.Add(subgraph.Key, new Subgraph(subgraph.Value, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        }

        var dfsAssignEdge = new QuikGraph.Algorithms.Search.EdgeDepthFirstSearchAlgorithm<Vertex, Edge>(Graph);
        dfsAssignEdge.TreeEdge += (edge) =>
        {
            var u = edge.Source;
            var v = edge.Target;

            var u_sub_idx = u.SubgraphIndex;
            var v_sub_idx = v.SubgraphIndex;

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

        // Find 0 input subgraphs and merge with output subgraph
        subgraph_rm_list.Clear();
        foreach (var sm in SubgraphMap)
        {
            var subgraph = sm.Value;
            if (subgraph.InputEdges.Count > 0)
            {
                continue;
            }

            var subgraphCompat = subgraph.Nodes.First().CompatType;
            HashSet<int> outSubgraphs = new();
            foreach (var edge in subgraph.OutputEdges)
            {
                var v = edge.Target;
                if (v.CompatType == subgraphCompat)
                {
                    outSubgraphs.Add(v.SubgraphIndex);
                }
            }

            if (outSubgraphs.Count > 1 || outSubgraphs.Count == 0)
            {
                continue;
            }

            var targetIdx = outSubgraphs.First();
            var targetSubgraph = SubgraphMap[targetIdx];

            sm.Value.Nodes.ForEach(x => x.SubgraphIndex = targetIdx);

            targetSubgraph.Nodes.AddRange(sm.Value.Nodes);
            targetSubgraph.InteriorEdges.AddRange(sm.Value.OutputEdges);
            targetSubgraph.InteriorEdges.AddRange(sm.Value.InteriorEdges);
            subgraph_rm_list.Add(sm.Key);
        }

        subgraph_rm_list.ForEach(x => SubgraphMap.Remove(x));
    }

    public void SummarizeGraph()
    {
        Dictionary<int, Vertex> indexMap = new();

        foreach (var subgraph in SubgraphMap)
        {
            var u = new Vertex(subgraph.Value.Nodes[0].Expr, subgraph.Value.Nodes[0].CompatType, subgraph.Key);
            indexMap.Add(subgraph.Key, u);
            GraphSummary.AddVertex(u);
        }

        Dictionary<Edge, Edge> edgeMap = new();
        foreach (var subgraph in SubgraphMap)
        {
            foreach (var edge in subgraph.Value.OutputEdges)
            {
                var u = indexMap[edge.Source.SubgraphIndex];
                var v = indexMap[edge.Target.SubgraphIndex];

                var newEdge = new Edge(edge.EdgeType, -1, u, v);
                GraphSummary.AddEdge(newEdge);
                if (edgeMap.ContainsKey(newEdge))
                {
                    System.Console.WriteLine("[ERROR] " + edge + " already mapped!");
                }

                edgeMap.Add(newEdge, edge);
            }
        }

        List<Edge> cycles = new();
        var dfs = new QuikGraph.Algorithms.Search.EdgeDepthFirstSearchAlgorithm<Vertex, Edge>(GraphSummary);
        dfs.BackEdge += (edge) =>
        {
            var u = edge.Source;
            var v = edge.Target;
            cycles.Add(edge);
        };
        dfs.Compute();

        bool modified = false;
        do
        {
            if (cycles.Count > 0)
            {
                var sumEdge = cycles.First();
                var edge = edgeMap[sumEdge];

                var u = edge.Source;
                var v = edge.Target;

                var nextSubgraphIdx = SubgraphMap.Last().Key + 1;
                var oldSubgraphIdx = v.SubgraphIndex;
                var oldSubgraph = SubgraphMap[oldSubgraphIdx];
                oldSubgraph.Nodes.Remove(v);

                v.SubgraphIndex = nextSubgraphIdx;
                SubgraphMap[nextSubgraphIdx] = new Subgraph(new List<Vertex>() { v }, new List<Edge>(), new List<Edge>(), new List<Edge>());

                foreach (var sm in SubgraphMap)
                {
                    sm.Value.InteriorEdges.Clear();
                    sm.Value.OutputEdges.Clear();
                    sm.Value.InputEdges.Clear();
                }

                var dfsAssignEdge = new QuikGraph.Algorithms.Search.EdgeDepthFirstSearchAlgorithm<Vertex, Edge>(Graph);
                dfsAssignEdge.TreeEdge += (edge) =>
                {
                    var u = edge.Source;
                    var v = edge.Target;

                    var u_sub_idx = u.SubgraphIndex;
                    var v_sub_idx = v.SubgraphIndex;

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

                modified = true;
                GraphSummary = new();
            }
        } while (modified);
    }
}

internal sealed class GraphConvertor : ExprVisitor<Unit, Unit, GraphContext>
{
    private int NodeCount { get; set; }

    protected override Unit DefaultVisitLeaf(Expr expr, GraphContext context)
    {
        Vertex target;
        if (expr is Call { Target: IR.CPU.Boxing } || expr.CheckedType is DistributedType)
        {
            target = new Vertex(expr, Compat.COMPATIBLE, -1);
        }
        else
        {
            target = new Vertex(expr, Compat.INCOMPATIBLE, -1);
        }

        context.Graph.AddVertex(target);
        context.SubgraphNodes.Add(NodeCount++, new() { target });
        foreach (var operand in expr.Operands)
        {
            if (context.Graph.Vertices.Any(v => v.Expr == operand))
            {
                var source = context.Graph.Vertices.First(v => v.Expr == operand);
                switch (source.CompatType, target.CompatType)
                {
                    case (Compat.COMPATIBLE, Compat.INCOMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.C2I, -1, source, target));
                        break;
                    case (Compat.INCOMPATIBLE, Compat.COMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.I2C, -1, source, target));
                        break;
                    case (Compat.INCOMPATIBLE, Compat.INCOMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.I2I, -1, source, target));
                        break;
                    default:
                        context.Graph.AddEdge(new Edge(EdgeTypes.C2C, -1, source, target));
                        break;
                }
            }
        }

        return Unit.Default;
    }
}
