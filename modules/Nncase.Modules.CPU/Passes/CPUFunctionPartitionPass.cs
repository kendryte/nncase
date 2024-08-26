// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes.GraphPartition;
using QuikGraph;

namespace Nncase.Passes;

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
                var ctx = new GraphPartition.GraphContext();
                var convertor = new GraphPartition.GraphConvertor(x => x switch
                {
                    Call call => (call.Target is IR.CPU.Boxing || call.CheckedType is DistributedType) ? true : false,
                    IR.Tuple tp => tp.Fields.ToArray().Any(f => f is Call { Target: IR.CPU.Boxing } b && b.CheckedType is TensorType) ? false : true,
                    _ => throw new NotSupportedException(),
                });
                convertor.Visit(pre.Body, ctx);

#if Debug
                ctx.Graph.DumpDot(DumpScope.Current.Directory + $"function_{i}.dot");
#endif

                ctx.SummarizeGraph();

#if Debug
                ctx.GraphSummary.DumpDot(DumpScope.Current.Directory + $"function_{i}_summary.dot");
#endif

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
                        var merger = new Rules.FusionMerger(ctx.VarMap[ctx.SummaryVertexSubgraphMap[vertex]]);
                        var clonedRoot = merger.Clone(vertex.Expr, default);

                        if (clonedRoot is IR.Tuple tuple)
                        {
                            clonedRoot = new IR.Tuple(tuple.Fields.AsValueEnumerable().Select(f => f.CheckedType is DistributedType d ? IR.F.CPU.Boxing(f, d.TensorType) : f).ToArray());
                        }

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
