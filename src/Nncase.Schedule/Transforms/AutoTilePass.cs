// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Passes.GraphPartition;
using Nncase.Schedule;
using QuikGraph;
using QuikGraph.Algorithms;

namespace Nncase.Passes.Transforms;

public sealed class AutoTilePass : ModulePass
{
    public AutoTilePass(string moduleKind, CompileOptions compileOptions)
    {
        ModuleKind = moduleKind;
        CompileOptions = compileOptions;
        WorkItem = 0;
    }

    public string ModuleKind { get; }

    public CompileOptions CompileOptions { get; }

    public int WorkItem { get; set; }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var tiler = new GraphTiler();
        var funcNums = input.Functions.Count;
        for (int i = 0; i < funcNums; i++)
        {
            var post = Rewrite(input.Functions[i], i, tiler);
            input.Replace(i, post);
        }

        return Task.FromResult(input);
    }

    private BaseFunction Rewrite(BaseFunction pre, int funcNumber, GraphTiler tiler)
    {
        if (!(pre is IR.Fusion fusion && fusion.ModuleKind == ModuleKind))
        {
            return pre;
        }

        var funcName = pre.Name;

        // 1. convert to quikgraph
        var graph = new BidirectionalGraph<ExprVertex, ExprEdge>(false);
        {
            var convertor = new AutoTileExprGraphConvertor();
            convertor.Visit(fusion.Body, graph);
        }

        // 2. perform condensation
        var condenseAlgo = new CondensationGraphAlgorithm<ExprVertex, ExprEdge>(graph);
        condenseAlgo.IsEdgeCompatible += (algo, arg) =>
        {
            return (arg.Edge.Source.Expr, arg.Edge.Target.Expr) switch
            {
                (Grid, Grid) => true,
                (Grid, IR.Tuple tp) => tp.Fields.AsValueEnumerable().All(x => x is Grid),
                _ => false,
            };
        };

        condenseAlgo.IsGraphCompatible += (algo, edge) =>
        {
            return algo.CondensedGraph.IsDirectedAcyclicGraph();
        };

        condenseAlgo.Compute();

        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Rewrite))
        {
            condenseAlgo.CondensedGraph.Dump($"{funcName}Condensed", init => { });
            condenseAlgo.ClusteredGraph.Dump($"{funcName}Cluster", algo =>
            {
                algo.FormatVertex += (s, arg) =>
                {
                    arg.VertexFormat.Label = $"{arg.Vertex.Expr.GetType().Name}";
                };
            });
        }

        // 3. reconstruction
        var constructor = new AutoTileReConstructor(tiler, funcNumber, ModuleKind, CompileOptions, condenseAlgo);
        var post = constructor.Construct();
        return fusion.With(fusion.Name, fusion.ModuleKind, post, fusion.Parameters.ToArray());
    }
}

internal sealed class AutoTileExprGraphConvertor : ExprGraphConvertor<ExprVertex, ExprEdge>
{
    protected override ExprVertex VisitGrid(Grid expr, IMutableVertexAndEdgeListGraph<ExprVertex, ExprEdge> context)
    {
        foreach (var read in expr.Reads)
        {
            Visit(read, context);
        }

        return VisitLeafGrid(expr, context);
    }

    protected override ExprVertex VisitLeafGrid(Grid expr, IMutableVertexAndEdgeListGraph<ExprVertex, ExprEdge> graph)
    {
        var target = (ExprVertex)ExprVertex.Create(expr);
        graph.AddVertex(target);
        int count = 0;
        foreach (var item in expr.Reads)
        {
            var source = Visit(item, graph);
            var edge = (ExprEdge)ExprEdge.Create(source, target, count++);
            graph.AddEdge(edge);
        }

        return target;
    }
}

internal sealed class AutoTileReConstructor : ExprReConstructor<ExprVertex, ExprEdge>
{
    public AutoTileReConstructor(GraphTiler tiler, int funcNumber, string moduleKind, CompileOptions compileOptions, CondensationGraphAlgorithm<ExprVertex, ExprEdge> algo)
        : base(algo)
    {
        Tiler = tiler;
        FuncNumber = funcNumber;
        ModuleKind = moduleKind;
        CompileOptions = compileOptions;
    }

    public GraphTiler Tiler { get; }

    public int FuncNumber { get; }

    public string ModuleKind { get; }

    public CompileOptions CompileOptions { get; }

    protected override Expr OnComplexCluster(ClusteredBidirectionalGraph<ExprVertex, ExprEdge> cluster, int sortIndex)
    {
        var pairs = GetClusterArgumentPairs(cluster);
        var extractDict = new Dictionary<Expr, Expr>(ReferenceEqualityComparer.Instance);
        var argumentDict = new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance);
        foreach (var (pre, post) in pairs)
        {
            if (pre is Const)
            {
                continue;
            }

            var @var = new Var(pre.CheckedType);
            var added = extractDict.TryAdd(pre, @var);
            if (added)
            {
                argumentDict.Add(@var, post);
            }
        }

        var cloner = new ExprClusterCloner(extractDict);
        var outVertices = cluster.OutVertices().ToArray();
        var clones = new List<Expr>();
        foreach (var outVertex in outVertices)
        {
            clones.Add(cloner.Clone(outVertex.Expr, default));
        }

        Expr cloned = clones.Count == 1 ? clones[0] : new IR.Tuple(clones.ToArray());
        var tiled = Tiler.Tile(cloned, ModuleKind, $"{FuncNumber}_{sortIndex}", (ICpuTargetOptions)CompileOptions.TargetOptions);

        var substitutor = new Mutators.Substitutor(e =>
        {
            if (e is Var v && argumentDict.TryGetValue(v, out var arg))
            {
                return arg;
            }

            return null;
        });

        var substited = substitutor.Rewrite(tiled, default);
        return substited;
    }
}
