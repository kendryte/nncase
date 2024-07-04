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
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using QuikGraph;
using QuikGraph.Algorithms;
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
    public Vertex(Expr expr, Compat compatType)
    {
        Expr = expr;
        CompatType = compatType;
    }

    public Expr Expr { get; set; }

    public Compat CompatType { get; set; }

    public override string ToString() => Expr.ToString();

    public QuikGraph.Graphviz.Dot.GraphvizColor Color() => CompatType switch
    {
        Compat.INCOMPATIBLE => QuikGraph.Graphviz.Dot.GraphvizColor.Coral,
        Compat.COMPATIBLE => QuikGraph.Graphviz.Dot.GraphvizColor.Olive,
        _ => QuikGraph.Graphviz.Dot.GraphvizColor.Cornsilk,
    };

    public bool Equals(Vertex? other)
    {
        if (other is null)
        {
            return false;
        }

        return ReferenceEquals(Expr, other.Expr) && EqualityComparer<Compat>.Default.Equals(CompatType, other.CompatType);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(ReferenceEqualityComparer.Instance.GetHashCode(Expr), CompatType.GetHashCode());
    }
}

public sealed record Edge : QuikGraph.IEdge<Vertex>
{
    public Edge(EdgeTypes edgeType, Vertex source, Vertex target)
    {
        EdgeType = edgeType;
        Source = source;
        Target = target;
    }

    public EdgeTypes EdgeType { get; set; }

    public Vertex Source { get; set; }

    public Vertex Target { get; set; }

    public bool Equals(Edge? other)
    {
        if (other is null)
        {
            return false;
        }

        return ReferenceEquals(Source, other.Source) &&
        ReferenceEquals(Target, other.Target) &&
        EqualityComparer<EdgeTypes>.Default.Equals(EdgeType, other.EdgeType);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(ReferenceEqualityComparer.Instance.GetHashCode(Source), ReferenceEqualityComparer.Instance.GetHashCode(Target), EdgeType.GetHashCode());
    }
}

public sealed class Graph : AdjacencyGraph<Vertex, Edge>
{
    public void DumpDot(string fullPathName)
    {
        using (var writer = new StreamWriter(fullPathName))
        {
            var a = this.ToGraphviz<Vertex, Edge>(algorithm =>
            {
                algorithm.FormatVertex += (_, args) => args.VertexFormat.Label = args.Vertex.ToString();
                algorithm.FormatVertex += (_, args) => args.VertexFormat.Style = QuikGraph.Graphviz.Dot.GraphvizVertexStyle.Filled;
                algorithm.FormatVertex += (_, args) => args.VertexFormat.FillColor = args.Vertex.Color();
            });
            writer.Write(a);
        }
    }
}

public sealed record Subgraph(int Index, List<Vertex> Nodes, List<Edge> InputEdges, List<Edge> OutputEdges, List<Edge> InteriorEdges);

public sealed class CPUFunctionPartitionPass : ModulePass
{
    public CPUFunctionPartitionPass(string moduleKind = "cpu")
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; set; }

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

                ctx.Graph.DumpDot(DumpScope.Current.Directory + $"function_{i}.dot");

                ctx.SummarizeGraph();

                ctx.GraphSummary.DumpDot(DumpScope.Current.Directory + $"function_{i}_summary.dot");

                var dfsVisitor = new QuikGraph.Algorithms.TopologicalSort.SourceFirstTopologicalSortAlgorithm<Vertex, Edge>(ctx.GraphSummary);
                dfsVisitor.Compute();
                var exprMemo = new Dictionary<Expr, Expr>(ReferenceEqualityComparer.Instance);
                for (var vi = 0; vi < dfsVisitor.SortedVertices.Length; vi++)
                {
                    var vertex = dfsVisitor.SortedVertices[vi];
                    var subgraph = ctx.SubgraphMap[ctx.SummaryVertexSubgraphMap[vertex]];
                    if (vertex.CompatType == Compat.INCOMPATIBLE)
                    {
                        var sg = new Graph();
                        subgraph.Nodes.ForEach(n => sg.AddVertex(n));
                        subgraph.InteriorEdges.ForEach(e => sg.AddEdge(e));

                        // sg.DumpDot(DumpScope.Current.Directory + $"_Incompatible_{subgraph.Index}.dot");
                        var sgVisitor = new QuikGraph.Algorithms.TopologicalSort.SourceFirstTopologicalSortAlgorithm<Vertex, Edge>(sg);
                        sgVisitor.Compute();
                        foreach (var v in sgVisitor.SortedVertices)
                        {
                            var expr = v.Expr switch
                            {
                                Call c => c.With(arguments: c.Arguments.AsValueEnumerable().Select(arg => exprMemo[arg]).ToArray()),
                                IR.Tuple t => t.With(fields: t.Fields.AsValueEnumerable().Select(arg => exprMemo[arg]).ToArray()),
                                _ => v.Expr,
                            };
                            exprMemo.Add(v.Expr, expr);
                        }
                    }
                    else
                    {
                        var newInputs = ctx.VarMap[ctx.SummaryVertexSubgraphMap[vertex]].Values.ToArray();
                        var merger = new Passes.Rules.FusionMerger(ctx.VarMap[ctx.SummaryVertexSubgraphMap[vertex]]);
                        var clonedRoot = merger.Clone(vertex.Expr, default);

                        var rootCall = new Call(new Fusion($"Function_{i}_fusion_{vi}_kernel", ModuleKind, clonedRoot, newInputs), ctx.VarMap[ctx.SummaryVertexSubgraphMap[vertex]].Keys.Select(e => exprMemo[e]).ToArray());
                        if (ctx.OutputMap[subgraph.Index].Count > 1)
                        {
                            ctx.OutputMap[subgraph.Index].ToList().ForEach(e => exprMemo.Add(e.Key, new Call(new GetItem(), rootCall, e.Value)));
                        }
                        else
                        {
                            exprMemo.Add(ctx.OutputMap[subgraph.Index].Keys.First(), rootCall);
                        }
                    }
                }

                var post = pre.With(pre.Name, pre.ModuleKind, exprMemo[pre.Body], pre.Parameters.ToArray());
                module.Replace(i, post);
            }
        }

        return Task.FromResult(module);
    }
}

internal sealed class GraphContext
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
        foreach (var subgraph in SubgraphMap)
        {
            var sg = new Graph();
            subgraph.Value.Nodes.ForEach(n => sg.AddVertex(n));
            subgraph.Value.InteriorEdges.ForEach(e => sg.AddEdge(e));

            // sg.DumpDot(DumpScope.Current.Directory + $"subgraph_{subgraph.Key}.dot");
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
                    var input = subgraph.Value.InputEdges.Find(e => e.Target == vertex)!.Source.Expr;
                    if (input is not Const && !VarMap[subgraph.Key].ContainsKey(input))
                    {
                        VarMap[subgraph.Key].Add(input, new Var(input.CheckedType));
                    }
                }
            }

            var u = new Vertex(null!, Compat.UNKNOWN);
            var outVertices = sg.Vertices.Count() == 1 ? sg.Vertices : sg.Edges.Where(e => !sg.OutEdges(e.Target).Any()).Select(e => e.Target).Distinct().ToList();
            u.CompatType = outVertices.First().CompatType;

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

internal sealed class GraphConvertor : ExprVisitor<Unit, Unit, GraphContext>
{
    private int NodeCount { get; set; }

    protected override Unit VisitLeafVar(Var expr, GraphContext context)
    {
        Vertex target;

        target = new Vertex(expr, Compat.INCOMPATIBLE);

        context.Graph.AddVertex(target);
        context.SubgraphMap.Add(NodeCount, new Subgraph(NodeCount, new() { target }, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        NodeCount++;

        return Unit.Default;
    }

    protected override Unit VisitLeafConst(Const expr, GraphContext context)
    {
        Vertex target;
        target = new Vertex(expr, expr.CheckedType is DistributedType ? Compat.COMPATIBLE : Compat.INCOMPATIBLE);

        context.Graph.AddVertex(target);
        context.SubgraphMap.Add(NodeCount, new Subgraph(NodeCount, new() { target }, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        NodeCount++;

        return Unit.Default;
    }

    protected override Unit VisitLeafCall(Call expr, GraphContext context)
    {
        Vertex target;
        if (expr.Target is IR.CPU.Boxing || expr.CheckedType is DistributedType)
        {
            target = new Vertex(expr, Compat.COMPATIBLE);
        }
        else
        {
            target = new Vertex(expr, Compat.INCOMPATIBLE);
        }

        context.Graph.AddVertex(target);
        context.SubgraphMap.Add(NodeCount, new Subgraph(NodeCount, new() { target }, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        NodeCount++;
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

        return Unit.Default;
    }

    protected override Unit VisitLeafTuple(IR.Tuple expr, GraphContext context)
    {
        Vertex target;
        var compatType = context.Graph.Vertices.FindFirst(v => ReferenceEquals(v.Expr, expr.Fields[0])).CompatType;
        target = new Vertex(expr, compatType);

        context.Graph.AddVertex(target);
        context.SubgraphMap.Add(NodeCount, new Subgraph(NodeCount, new() { target }, new List<Edge>(), new List<Edge>(), new List<Edge>()));
        NodeCount++;
        foreach (var field in expr.Fields)
        {
            if (context.Graph.Vertices.Any(v => ReferenceEquals(v.Expr, field)))
            {
                var source = context.Graph.Vertices.First(v => ReferenceEquals(v.Expr, field));
                switch (source.CompatType, target.CompatType)
                {
                    case (Compat.INCOMPATIBLE, Compat.INCOMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.I2I, source, target));
                        break;
                    case (Compat.COMPATIBLE, Compat.COMPATIBLE):
                        context.Graph.AddEdge(new Edge(EdgeTypes.C2C, source, target));
                        break;
                    default:
                        throw new InvalidOperationException("Not Supported Compat Type");
                }
            }
        }

        return Unit.Default;
    }

    protected override Unit DefaultVisitLeaf(Expr expr, GraphContext context)
    {
        return Unit.Default;
    }
}
