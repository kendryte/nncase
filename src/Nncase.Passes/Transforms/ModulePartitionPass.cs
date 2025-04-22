// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes.GraphPartition;
using Nncase.Targets;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Graphviz;

namespace Nncase.Passes.Transforms;

public sealed class ModulePartitionPass : ModulePass
{
    public ModulePartitionPass(IModuleCompiler moduleCompiler)
    {
        ModuleCompiler = moduleCompiler;
    }

    public IModuleCompiler ModuleCompiler { get; }

    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext context)
    {
        var funcs = module.Functions.Count;
        for (int i = 0; i < funcs; i++)
        {
            if (module.Functions[i] is not Function function)
            {
                continue;
            }

            Function pre = function;
            var postBody = PerformPartition(module, pre.Name, pre.Body);
            var post = pre.With(pre.Name, pre.ModuleKind, postBody, pre.Parameters.ToArray());
            module.Replace(i, post);
        }

        return Task.FromResult(module);
    }

    private Expr PerformPartition(IRModule module, string funcName, Expr pre)
    {
        var dynamicVars = IRHelpers.GetDynamicDimVars();

        // 1. convert to quikgraph
        var biGraph = new BidirectionalGraph<ExprVertex, ExprEdge>(false);
        {
            var graphConvertor = new ExprGraphConvertor<ExprVertex, ExprEdge>();
            graphConvertor.Visit(pre, biGraph);
        }

        // 2. perform condensation
        var condenseAlgo = new CondensationGraphAlgorithm<ExprVertex, ExprEdge>(biGraph);
        condenseAlgo.IsEdgeCompatible += (algo, arg) =>
        {
            bool CheckField(Expr expr)
            {
                return expr switch
                {
                    Const => true,
                    Call call => ModuleCompiler.IsSupportedCall(call, CompileSession.CompileOptions),
                    _ => false,
                };
            }

            bool isSupport = false;
            switch (arg.Edge.Source.Expr, arg.Edge.Target.Expr)
            {
                case (Call callee, Call caller):
                    isSupport = CheckField(callee) && CheckField(caller);
                    break;
                case (IR.Tuple tuple, Call caller):
                    isSupport = tuple.Fields.AsValueEnumerable().All(CheckField) && CheckField(caller);
                    break;
                case (Call callee, IR.Tuple tuple):
                    isSupport = CheckField(callee) && tuple.Fields.AsValueEnumerable().All(CheckField);
                    break;
                default:
                    break;
            }

            return isSupport;
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
        var constructor = new DistributedReconstructor(module, funcName, ModuleCompiler, CompileSession.CompileOptions, condenseAlgo);
        var post = constructor.Construct();
        return post;
    }
}

internal sealed class DistributedReconstructor : ExprReconstructor<ExprVertex, ExprEdge>
{
    public DistributedReconstructor(IRModule module, string funcName, IModuleCompiler moduleCompiler, CompileOptions compileOptions, CondensationGraphAlgorithm<ExprVertex, ExprEdge> algo)
        : base(algo)
    {
        Module = module;
        FuncName = funcName;
        ModuleCompiler = moduleCompiler;
        CompileOptions = compileOptions;
    }

    public IRModule Module { get; }

    public string FuncName { get; }

    public IModuleCompiler ModuleCompiler { get; }

    public CompileOptions CompileOptions { get; }

    protected override Expr OnAtomCluster(ClusteredBidirectionalGraph<ExprVertex, ExprEdge> cluster, int sortIndex)
    {
        if (cluster.Vertices.First().Expr is Call call && ModuleCompiler.IsSupportedCall(call, CompileOptions))
        {
            return OnComplexCluster(cluster, sortIndex);
        }

        return base.OnAtomCluster(cluster, sortIndex);
    }

    protected override Expr OnComplexCluster(ClusteredBidirectionalGraph<ExprVertex, ExprEdge> cluster, int sortIndex)
    {
        var pairs = GetClusterArgumentPairs(cluster);
        var processedParams = new HashSet<Expr>(ReferenceEqualityComparer.Instance);
        var extractDict = new Dictionary<Expr, Expr>(ReferenceEqualityComparer.Instance);
        var @params = new List<Var>();
        var arguments = new List<Expr>();
        var dynamicVars = IRHelpers.GetDynamicDimVars();

        foreach (var dimVar in dynamicVars)
        {
            if (processedParams.Add(dimVar))
            {
                extractDict.Add(dimVar, dimVar);
                @params.Add(dimVar);
                arguments.Add(dimVar);
            }
        }

        foreach (var (pre, post) in pairs)
        {
            if (pre is not (Call or Var or If))
            {
                continue;
            }

            Var @var;
            Expr extract;
            if (pre is Var preVar && dynamicVars.Contains(preVar))
            {
                continue;
            }
            else if (pre.CheckedType is DistributedType d)
            {
                @var = pre is Var oldVar ? oldVar.With(typeAnnotation: d.TensorType) : new Var(d.TensorType) { Metadata = pre.Metadata };
                extract = IR.F.Distributed.Boxing(@var, d);
            }
            else if (pre.CheckedType is TupleType tupleType)
            {
                if (processedParams.Add(pre))
                {
                    var extractFields = new Expr[tupleType.Fields.Count];
                    for (int i = 0; i < tupleType.Fields.Count; i++)
                    {
                        var field = tupleType.Fields[i];

                        if (dynamicVars.Contains(((IR.Tuple)pre).Fields[i]))
                        {
                            // can't skip, extractFields will miss some var.
                            @var = (Var)((IR.Tuple)pre).Fields[i];
                        }
                        else if (((IR.Tuple)pre).Fields[i] is TensorConst tensorConst)
                        {
                            extractFields[i] = tensorConst;
                            continue;
                        }
                        else
                        {
                            @var = new Var(field) { Metadata = pre.Metadata };
                            @params.Add(@var);
                            arguments.Add(post[i]);
                        }

                        extractFields[i] = @var;
                    }

                    extractDict.Add(pre, new IR.Tuple(extractFields));
                }

                continue;
            }
            else
            {
                @var = pre is Var oldVar ? oldVar.With() : new Var(pre.CheckedType) { Metadata = pre.Metadata };
                extract = @var;
            }

            if (processedParams.Add(pre))
            {
                extractDict.Add(pre, extract);
                @params.Add(@var);
                arguments.Add(post);
            }
        }

        var cloner = new ExprClusterCloner(extractDict);
        var outVertices = cluster.OutVertices(Algo.ClusteredGraph).ToArray();
        var clones = new List<Expr>();
        foreach (var outVertex in outVertices)
        {
            clones.Add(cloner.Clone(outVertex.Expr, default));
        }

        var cloned = PostProcess(clones);
        var func = new Function($"{FuncName}_{sortIndex}_kernel", ModuleCompiler.ModuleKind, cloned, @params.ToArray());
        Module.Add(func);
        return new Call(func, arguments.ToArray());
    }

    private Expr PostProcess(List<Expr> clones)
    {
        Expr PostProcessSingle(Expr cloned, out bool changed)
        {
            changed = false;
            switch (cloned)
            {
                case IR.Tuple tp:
                    var nFields = new List<Expr>();
                    foreach (var item in tp.Fields)
                    {
                        nFields.Add(PostProcessSingle(item, out var childChanged));
                        changed |= childChanged;
                    }

                    if (changed)
                    {
                        return new IR.Tuple(nFields.ToArray());
                    }
                    else
                    {
                        return tp;
                    }

                case Expr e when e.CheckedType is DistributedType d:
                    changed = true;
                    return IR.F.Distributed.Boxing(e, d.TensorType);
                default:
                    return cloned;
            }
        }

        if (clones.Count == 1)
        {
            return PostProcessSingle(clones[0], out _);
        }
        else
        {
            return new IR.Tuple(clones.Select(c => PostProcessSingle(c, out _)).ToArray());
        }
    }
}
