// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using NetFabric.Hyperlinq;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes.GraphPartition;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Graphviz;

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
            if (module.Functions[i] is not Function function)
            {
                continue;
            }

            Function pre = function;
            var postBody = PerformPartition(pre.Name, pre.Body);
            var post = pre.With(pre.Name, pre.ModuleKind, postBody, pre.Parameters.ToArray());
            module.Replace(i, post);
        }

        return Task.FromResult(module);
    }

    private Expr PerformPartition(string funcName, Expr pre)
    {
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
            bool CheckField(Expr f)
            {
                if (f is Call c && c.Target is IR.CPU.Boxing { NewType: TensorType } && c.Arguments[0].CheckedType is DistributedType)
                {
                    return true;
                }

                return f.CheckedType is DistributedType;
            }

            bool isSupport = false;
            switch (arg.Edge.Target.Expr)
            {
                case Call call:
                    if (call.Target is IR.CPU.Boxing { NewType: TensorType } && call.Arguments[0].CheckedType is DistributedType)
                    {
                        isSupport = true;
                    }
                    else if (call.Target is IR.CPU.Boxing { NewType: DistributedType })
                    {
                        if (arg.Edge.Source.Expr.CheckedType is not TensorType)
                        {
                            isSupport = true;
                        }
                    }
                    else if (call.CheckedType is DistributedType)
                    {
                        isSupport = true;
                    }

                    break;
                case IR.Tuple tp:
                    isSupport = tp.Fields.AsValueEnumerable().All(f => f is Call c && CheckField(c)) ? true : false;
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
        var constructor = new DistributedReConstructor(funcName, ModuleKind, condenseAlgo);
        var post = constructor.Construct();
        return post;
    }
}

internal sealed class DistributedReConstructor : ExprReConstructor<ExprVertex, ExprEdge>
{
    public DistributedReConstructor(string funcName, string moduleKind, CondensationGraphAlgorithm<ExprVertex, ExprEdge> algo)
        : base(algo)
    {
        FuncName = funcName;
        ModuleKind = moduleKind;
    }

    public string FuncName { get; }

    public string ModuleKind { get; }

    protected override Expr OnComplexCluster(ClusteredBidirectionalGraph<ExprVertex, ExprEdge> cluster, int sortIndex)
    {
        var pairs = GetClusterArgumentPairs(cluster);
        var paramDict = new Dictionary<Expr, Var>(ReferenceEqualityComparer.Instance);
        var extractDict = new Dictionary<Expr, Expr>(ReferenceEqualityComparer.Instance);
        var argumentDict = new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance);
        foreach (var (pre, post) in pairs)
        {
            if (pre is Const)
            {
                continue;
            }

            Var @var;
            Expr extract;
            if (pre.CheckedType is DistributedType d)
            {
                @var = new Var(d.TensorType);
                extract = IR.F.CPU.Boxing(@var, d);
            }
            else
            {
                @var = new Var(pre.CheckedType);
                extract = @var;
            }

            var added = paramDict.TryAdd(pre, @var);
            if (added)
            {
                extractDict.Add(pre, extract);
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

        var cloned = PostProcess(clones);
        var fusion = new Fusion($"{FuncName}_fusion_{sortIndex}", ModuleKind, cloned, paramDict.Values.OfType<Var>().ToArray());
        return new Call(fusion, paramDict.Values.OfType<Var>().Select(v => argumentDict[v]).ToArray());
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
                    return IR.F.CPU.Boxing(e, d.TensorType);
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
