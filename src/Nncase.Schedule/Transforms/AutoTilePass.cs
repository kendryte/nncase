﻿// Copyright (c) Canaan Inc. All rights reserved.
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

public sealed class AutoTilePass : FunctionPass
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

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        var tiler = new GraphTiler();
        if (!(input is Function func && func.ModuleKind == ModuleKind))
        {
            return Task.FromResult(input);
        }

        var funcName = func.Name;

        // 1. convert to quikgraph
        var graph = new BidirectionalGraph<ExprVertex, ExprEdge>(false);
        {
            var convertor = new AutoTileExprGraphConvertor();
            convertor.Visit(func.Body, graph);
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
            condenseAlgo.CondensedGraph.Dump($"Condensed", init => { });
            condenseAlgo.ClusteredGraph.Dump($"Cluster", algo =>
            {
                algo.FormatVertex += (s, arg) =>
                {
                    arg.VertexFormat.Label = $"{arg.Vertex.Expr.GetType().Name}";
                };
            });
        }

        // 3. reconstruction
        var constructor = new AutoTileReconstructor(tiler, ModuleKind, CompileOptions, condenseAlgo);
        var post = constructor.Construct();
        return Task.FromResult((BaseFunction)func.With(body: post));
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

internal sealed class AutoTileReconstructor : ExprReconstructor<ExprVertex, ExprEdge>
{
    public AutoTileReconstructor(GraphTiler tiler, string moduleKind, CompileOptions compileOptions, CondensationGraphAlgorithm<ExprVertex, ExprEdge> algo)
        : base(algo)
    {
        Tiler = tiler;
        ModuleKind = moduleKind;
        CompileOptions = compileOptions;
    }

    public GraphTiler Tiler { get; }

    public string ModuleKind { get; }

    public CompileOptions CompileOptions { get; }

    protected override BaseExpr OnAtomCluster(ClusteredBidirectionalGraph<ExprVertex, ExprEdge> cluster, int sortIndex)
    {
        using var subscope = new Diagnostics.DumpScope($"cluster_{sortIndex}", Diagnostics.DumpFlags.Tiling);
        var pairs = GetClusterArgumentPairs(cluster);
        var vertex = cluster.Vertices.First();
        var expr = vertex.Expr;
        if (expr is Grid)
        {
            var extractDict = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
            var argumentDict = new Dictionary<Var, BaseExpr>(ReferenceEqualityComparer.Instance);
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
            var cloned = (Expr)cloner.Clone(expr, default);
            var tiled = Tiler.Tile(cloned, ModuleKind, (INTTTargetOptions)CompileOptions.TargetOptions);
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
        else
        {
            var cloner = new ExprClusterCloner(pairs.ToDictionary(p => p.Pre, p => p.Post, new ReferenceEqualityComparer<BaseExpr>()));
            return cloner.Clone(expr, default);
        }
    }

    protected override BaseExpr OnComplexCluster(ClusteredBidirectionalGraph<ExprVertex, ExprEdge> cluster, int sortIndex)
    {
        using var subscope = new Diagnostics.DumpScope($"cluster_{sortIndex}", Diagnostics.DumpFlags.Tiling);
        var pairs = GetClusterArgumentPairs(cluster);
        var extractDict = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
        var argumentDict = new Dictionary<Var, BaseExpr>(ReferenceEqualityComparer.Instance);
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

        // todo sometimes internal grid have outside dependence, so we can't fuse it when tiling.
        var cloner = new ExprClusterCloner(extractDict);
        var outVertices = cluster.OutVertices(Algo.ClusteredGraph).ToArray();
        var clones = new List<BaseExpr>();
        foreach (var outVertex in outVertices)
        {
            clones.Add(cloner.Clone(outVertex.Expr, default));
        }

        var cloned = clones.Count == 1 ? clones[0] : new IR.Tuple(clones.ToArray());
        var tiled = Tiler.Tile(cloned, ModuleKind, (INTTTargetOptions)CompileOptions.TargetOptions);
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
