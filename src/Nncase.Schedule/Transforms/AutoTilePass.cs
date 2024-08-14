// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Passes.GraphPartition;
using Nncase.Schedule;

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
        var funcNums = input.Functions.Count;
        for (int i = 0; i < funcNums; i++)
        {
            var post = Rewrite(input.Functions[i], i);
            input.Replace(i, post);
        }

        return Task.FromResult(input);
    }

    private BaseFunction Rewrite(BaseFunction pre, int i)
    {
        if (!(pre is IR.Fusion fusion && fusion.ModuleKind == ModuleKind))
        {
            return pre;
        }

        // Function post;
        var ctx = new GraphContext();
        var convertor = new GraphConvertor(x => x switch
        {
            Grid => true,
            IR.Tuple tp => tp.Fields.AsValueEnumerable().All(f => f is Grid),
            _ => false,
        });
        convertor.Visit(fusion.Body, ctx);

        ctx.SummarizeGraph();

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

                // sg.DumpDot(DumpScope.Current.Directory + $"_Incompatible_{subgraph.Index}_{vi}.dot");
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
                var merger = new ReplacingExprCloner(ctx.VarMap[ctx.SummaryVertexSubgraphMap[vertex]].ToDictionary(kv => kv.Key, kv => (Expr)kv.Value));
                var clonedRoot = merger.Clone(vertex.Expr, default);

                var newCall = TreeTiler.Tile((Grid)clonedRoot, ModuleKind, vi, CompileOptions.TargetOptions);

                exprMemo.Add(ctx.OutputMap[subgraph.Index].Keys.First(), newCall);
            }
        }

        return fusion.With(fusion.Name, fusion.ModuleKind, exprMemo[fusion.Body], fusion.Parameters.ToArray());
    }
}
