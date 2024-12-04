// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reactive;
using System.Runtime.CompilerServices;
using Google.OrTools.Sat;
using NetFabric.Hyperlinq;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.Targets;
using Nncase.Utilities;
using QuikGraph;
using QuikGraph.Graphviz;

[assembly: InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.Passes.Distributed;

internal enum SearchGraphKind : int
{
    Root,
    DistributedCluster,
    StandaloneCluster,
    Bucket,
}

/// <summary>
/// auto distributed the xpu fusion.
/// </summary>
[RuleGenerator]
public sealed partial class AutoDistributedPass : FunctionPass
{
    private readonly CompileOptions _compileOptions;

    public AutoDistributedPass(CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;
    }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        var rewriter = new AutoDistributedRewriter(_compileOptions, _compileOptions.TargetOptions is CpuTargetOptions options ? options : new CpuTargetOptions());
        return Task.FromResult(rewriter.Rewirte(input));
    }
}

internal sealed class SearchableNode
{
    public SearchableNode(Expr expr, IRType type)
    {
        Expr = expr;
        IRType = type;
    }

    public Expr Expr { get; }

    public IRType IRType { get; }
}

internal sealed record CrossEdge(SearchableNode Source, SearchableNode Target, int SourceIndex, DistributedSearchGraph TargetGraph) : IEdge<SearchableNode>
{
}

internal sealed class DistributedSearchGraph : TieredAdjacencyGraph<SearchableNode, CrossEdge>
{
    public DistributedSearchGraph([NotNull] AdjacencyGraph<SearchableNode, CrossEdge> wrappedGraph, SearchGraphKind kind)
    : base(wrappedGraph)
    {
        Kind = kind;
    }

    public DistributedSearchGraph([NotNull] TieredAdjacencyGraph<SearchableNode, CrossEdge> parentGraph, SearchGraphKind kind)
        : base(parentGraph)
    {
        Kind = kind;
    }

    public SearchGraphKind Kind { get; }
}

internal sealed class AutoDistributedRewriter : ExprVisitor<Unit, Unit>
{
    private readonly Dictionary<Expr, DistributedSearchGraph> _reshardMemo;

    private readonly Dictionary<Expr, DistributedSearchGraph> _inferedMemo;

    private readonly AdjacencyGraph<SearchableNode, CrossEdge> _rootGraph;

    private readonly DistributedSearchGraph _rootSearchGraph;

    public AutoDistributedRewriter(CompileOptions compileOptions, CpuTargetOptions targetOptions)
    {
        Placements = targetOptions.Hierarchies.Select(h => new Placement(h, targetOptions.HierarchyNames)).ToArray();
        CompileOptions = compileOptions;
        TargetOptions = targetOptions;
        if (Path.Exists(TargetOptions.DistributedScheme) && System.Text.Json.JsonSerializer.Deserialize<DistributedScheme>(File.ReadAllText(TargetOptions.DistributedScheme)) is DistributedScheme scheme)
        {
            Scheme = scheme.Outputs.ToDictionary(n => n.Name, n => (new IRArray<SBP>(n.NdSBP), new Placement(n.Hierarchy, n.HierarchyName)));
        }
        else
        {
            Scheme = new Dictionary<string, (IRArray<SBP> NdSBP, Placement Placement)>();
        }

        _reshardMemo = new(ReferenceEqualityComparer.Instance);
        _inferedMemo = new(ReferenceEqualityComparer.Instance);
        _rootGraph = new(false);
        _rootSearchGraph = new(_rootGraph, SearchGraphKind.Root);
    }

    public IRArray<Placement> Placements { get; }

    public CompileOptions CompileOptions { get; }

    public CpuTargetOptions TargetOptions { get; }

    public IReadOnlyDictionary<string, (IRArray<SBP> NdSBP, Placement Placement)> Scheme { get; }

    public static void MemoryExtractConstrains(CpModel model, IReadOnlyDictionary<ENode, BoolVar> vars)
    {
        var consts = vars.Keys.Where(k => k.Expr is Call { Target: IR.CPU.Boxing { NewType: DistributedType } } call && call.Arguments[0] is TensorConst tc && tc.Value.Length >= 8).ToArray();
        model.Add(LinearExpr.WeightedSum(consts.Select(k => vars[k]), consts.Select(k =>
        {
            var type = DistributedUtility.GetDividedTensorType((DistributedType)k.Expr.CheckedType);
            return TensorUtilities.GetProduct(type.Shape.ToValueArray()) * type.DType.SizeInBytes;
        })) < (2L * 512L * 1024L * 1024L));
    }

    public static IReadOnlyList<DistributedType> GetLeafCandidateDistTypes(TensorType tensorType, IEnumerable<Placement> placements)
    {
        return placements.Select(placement => DistributedUtility.GetLeafCandidateNDSBPs(tensorType, placement).Select(ndsbp => new DistributedType(tensorType, ndsbp, placement))).SelectMany(e => e).ToArray();
    }

    public void SingleNodeMemoryExtractConstrains(CpModel model, IReadOnlyDictionary<ENode, BoolVar> vars)
    {
        var distTypes = vars.Keys.Where(k => k.Expr.CheckedType is DistributedType dt).ToArray();
        foreach (var k in distTypes)
        {
            if (TargetOptions.HierarchySizes.Length > 1)
            {
                var type = DistributedUtility.GetDividedTensorType((DistributedType)k.Expr.CheckedType);
                var size = TensorUtilities.GetProduct(type.Shape.ToValueArray()) * type.DType.SizeInBytes;

                if (k.Expr is Call { Target: IR.CPU.Boxing boxing } call && boxing.NewType is DistributedType distributedType && call.Arguments[0].CheckedType is DistributedType inType && inType.NdSBP.Any(sbp => sbp is SBPPartialSum) && distributedType != call.Arguments[0].CheckedType)
                {
                    type = DistributedUtility.GetDividedTensorType(inType);
                    size += TensorUtilities.GetProduct(type.Shape.ToValueArray()) * type.DType.SizeInBytes;
                }

                model.Add(vars[k] * size < TargetOptions.HierarchySizes[^2] / TargetOptions.Hierarchies[0][^1]);
            }
        }
    }

    public void FilterByScheme(Expr expr, DistributedSearchGraph cluster)
    {
        foreach (var name in expr.Metadata.OutputNames ?? Array.Empty<string>())
        {
            if (Scheme.TryGetValue(name, out var tp))
            {
                if (cluster.Kind is SearchGraphKind.DistributedCluster)
                {
                    var removes = new List<DistributedSearchGraph>();
                    foreach (var bucket in cluster.Clusters.OfType<DistributedSearchGraph>())
                    {
                        bucket.RemoveVertexIf(v => !(v.IRType is DistributedType dtype && dtype.NdSBP == tp.NdSBP && dtype.Placement == tp.Placement));
                        if (bucket.VertexCount == 0)
                        {
                            removes.Add(bucket);
                        }
                    }

                    foreach (var r in removes)
                    {
                        cluster.RemoveCluster(r);
                    }
                }
            }
        }
    }

    public BaseFunction Rewirte(BaseFunction input)
    {
        if (input is Function { Body: Expr body } function)
        {
            Visit(body);
            var rootCluster = TryInstertTerminator(body);

            using (var stream = Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.PassIR) ? Diagnostics.DumpScope.Current.OpenFile("DistributedSearchGraph.dot") : Stream.Null)
            {
                Dump(stream, new Dictionary<SearchableNode, bool>() { });
            }

            var post = SolveAndExtract(rootCluster);
            return function.With(body: post);
        }

        return input;
    }

    protected override Unit DefaultVisitLeaf(Expr expr)
    {
        return default;
    }

    protected override Unit VisitLeafCall(Call expr)
    {
        if (expr.Target is not Op op)
        {
            throw new NotSupportedException();
        }

        var isSupported = PassUtility.IsCpuSupported(op, expr, expr.Arguments.ToArray());
        var callCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(isSupported ? SearchGraphKind.DistributedCluster : SearchGraphKind.StandaloneCluster);
        var argClusters = new DistributedSearchGraph[op.Parameters.Count()];
        foreach (var param in op.Parameters)
        {
            argClusters[param.Index] = VisitLeafArgument(param.ParameterKind, expr.Arguments[param.Index], isSupported);
        }

        var bucketMemo = new Dictionary<IRType, DistributedSearchGraph>();
        foreach (var combBuckets in argClusters.Select(c => c.Clusters.OfType<DistributedSearchGraph>()).CartesianProduct())
        {
            var tempArgs = combBuckets.Select<DistributedSearchGraph, Expr>(bucket => bucket.Vertices.First() switch
            {
                SearchableNode { Expr: Const attr } => attr,
                SearchableNode n => new Var(n.IRType),
            }).ToArray();
            var checkType = new Call(op, tempArgs).CheckedType;
            if (checkType is not DistributedType ndistType)
            {
                continue;
            }

            if (!bucketMemo.TryGetValue(checkType, out var dbucket))
            {
                dbucket = callCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                bucketMemo.Add(checkType, dbucket);
            }

            var dnode = new SearchableNode(op, ndistType);
            dbucket.AddVertex(dnode);

            foreach (var (arg, i) in combBuckets.Select((arg, i) => (arg, i)))
            {
                _rootSearchGraph.AddEdge(new(dnode, arg.Vertices.First(), i, arg));
            }
        }

        _inferedMemo.Add(expr, callCluster);

        if (!isSupported)
        {
            return default;
        }

        // expand the search space.
        var addedBuckets = bucketMemo.Values.ToArray();
        foreach (var candType in GetLeafCandidateDistTypes(expr.CheckedTensorType, Placements))
        {
            if (!bucketMemo.TryGetValue(candType, out var bucket))
            {
                bucket = callCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var dnode = new SearchableNode(new Boxing(candType, false), candType);
                bucket.AddVertex(dnode);

                foreach (var added in addedBuckets)
                {
                    callCluster.AddEdge(new(dnode, added.Vertices.First(), 0, added));
                }

                bucketMemo.Add(candType, bucket);
            }
        }

        FilterByScheme(expr, callCluster);
        return default;
    }

    private DistributedSearchGraph VisitLeafArgument(ParameterKind parameterKind, Expr expr, bool isSupported)
    {
        DistributedSearchGraph TryInsertNonDistStarter(Expr e)
        {
            if (!_inferedMemo.TryGetValue(e, out var attrCluster))
            {
                attrCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.StandaloneCluster);
                var bucketGraph = attrCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                bucketGraph.AddVertex(new SearchableNode(e, e.CheckedType));
                _inferedMemo.Add(e, attrCluster);
            }

            return attrCluster;
        }

        DistributedSearchGraph? argCluster = null;
        switch (parameterKind, expr)
        {
            case (ParameterKind.Input, Expr e) when e is Const or Var:
                if (isSupported)
                {
                    argCluster = TryAddOriginator(expr);
                }
                else
                {
                    argCluster = TryInsertNonDistStarter(e);
                }

                break;
            case (ParameterKind.Input, IR.Tuple tp):
                if (isSupported)
                {
                    if (!_inferedMemo.TryGetValue(tp, out var inferCluster))
                    {
                        inferCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
                        var fClusters = new DistributedSearchGraph[tp.Fields.Length];
                        foreach (var (f, i) in tp.Fields.AsValueEnumerable().Select((f, i) => (f, i)))
                        {
                            fClusters[i] = VisitLeafArgument(ParameterKind.Input, f, isSupported);
                        }

                        var bucketMemo = new Dictionary<TupleType, DistributedSearchGraph>();
                        foreach (var combBuckets in fClusters.Select(c => c.Clusters.OfType<DistributedSearchGraph>()).CartesianProduct())
                        {
                            var tpType = new TupleType(combBuckets.Select(b => b.Vertices.First().IRType).ToArray());
                            var tpNode = new SearchableNode(new IR.Tuple(), tpType);
                            if (!bucketMemo.TryGetValue(tpType, out var bucket))
                            {
                                bucket = inferCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                                bucketMemo.Add(tpType, bucket);
                            }

                            bucket.AddVertex(tpNode);
                            foreach (var (fbucket, i) in combBuckets.Select((b, i) => (b, i)))
                            {
                                _rootSearchGraph.AddEdge(new(tpNode, fbucket.Vertices.First(), i, fbucket));
                            }
                        }

                        _inferedMemo.Add(tp, inferCluster);
                    }

                    if (inferCluster.Kind != SearchGraphKind.DistributedCluster)
                    {
                        throw new NotSupportedException("tuple infer cluster is not dist cluster");
                    }

                    argCluster = inferCluster;
                }
                else
                {
                    throw new NotSupportedException("not support tuple input.");
                }

                break;
            case (ParameterKind.Input, Expr e):
                if (isSupported)
                {
                    argCluster = _inferedMemo[e];
                    if (argCluster.Kind is SearchGraphKind.StandaloneCluster)
                    {
                        argCluster = TryAddOriginator(e);
                    }
                }
                else
                {
                    argCluster = TryInstertTerminator(e);
                }

                break;
            case (ParameterKind.Attribute, Expr e):
                argCluster = TryInsertNonDistStarter(e);
                break;
            default:
                throw new InvalidOperationException();
        }

        FilterByScheme(expr, argCluster);
        return argCluster ?? throw new InvalidOperationException("the argument cluster can't be null.");
    }

    private bool IsDistributed(IRType type) => type switch
    {
        DistributedType => true,
        TupleType t => t.All(IsDistributed),
        _ => false,
    };

    private DistributedSearchGraph TryAddOriginator(Expr expr)
    {
        if (_reshardMemo.TryGetValue(expr, out var reshardGraph))
        {
            if (reshardGraph.Kind != SearchGraphKind.DistributedCluster)
            {
                throw new NotSupportedException();
            }

            return reshardGraph;
        }

        reshardGraph = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
        _reshardMemo.Add(expr, reshardGraph);

        if (expr is IR.Tuple tp)
        {
            var buckets = new DistributedSearchGraph[tp.Fields.Length];
            foreach (var (f, fGraph, i) in tp.Fields.AsValueEnumerable().Select((f, i) => (f, Visit(f), i)))
            {
                buckets[i] = TryAddOriginator(f).Clusters.OfType<DistributedSearchGraph>().First();
            }

            var tpnode = new SearchableNode(new IR.Tuple(), new TupleType(buckets.Select(g => g.Vertices.First().IRType).ToArray()));
            var bucket = reshardGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            bucket.AddVertex(tpnode);
            for (int i = 0; i < tp.Fields.Length; i++)
            {
                _rootSearchGraph.AddEdge(new(tpnode, buckets[i].Vertices.First(), i, buckets[i]));
            }
        }
        else
        {
            if (!_inferedMemo.TryGetValue(expr, out var inferCluster))
            {
                var onode = new SearchableNode(expr, expr.CheckedType);
                inferCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.StandaloneCluster);
                var obucket = inferCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                obucket.AddVertex(onode);
                _inferedMemo.Add(expr, inferCluster);
            }

            foreach (var dType in GetLeafCandidateDistTypes((TensorType)inferCluster.Vertices.First().IRType, Placements))
            {
                var bucket = reshardGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var dnode = new SearchableNode(new Boxing(dType, false), dType);
                bucket.AddVertex(dnode);
                _rootSearchGraph.AddEdge(new(dnode, inferCluster.Vertices.First(), 0, inferCluster.Clusters.OfType<DistributedSearchGraph>().First()));
            }
        }

        return reshardGraph;
    }

    private DistributedSearchGraph TryInstertTerminator(Expr expr)
    {
        if (_reshardMemo.TryGetValue(expr, out var reshardGraph))
        {
            if (reshardGraph.Kind != SearchGraphKind.StandaloneCluster)
            {
                throw new NotSupportedException();
            }

            return reshardGraph;
        }

        reshardGraph = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.StandaloneCluster);

        if (expr is IR.Tuple tp)
        {
            var buckets = new DistributedSearchGraph[tp.Fields.Length];
            foreach (var (f, fGraph, i) in tp.Fields.AsValueEnumerable().Select((f, i) => (f, Visit(f), i)))
            {
                buckets[i] = TryInstertTerminator(f).Clusters.OfType<DistributedSearchGraph>().First();
            }

            var tpnode = new SearchableNode(new IR.Tuple(), new TupleType(buckets.Select(g => g.Vertices.First().IRType).ToArray()));
            var bucket = reshardGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            bucket.AddVertex(tpnode);
            for (int i = 0; i < tp.Fields.Length; i++)
            {
                _rootSearchGraph.AddEdge(new(tpnode, buckets[i].Vertices.First(), i, buckets[i]));
            }
        }
        else
        {
            var onode = new SearchableNode(new Boxing(expr.CheckedType, false), expr.CheckedType);
            var inputBuckets = _inferedMemo[expr].Clusters.OfType<DistributedSearchGraph>().ToArray();

            var bucket = reshardGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            bucket.AddVertex(onode);
            foreach (var inputBucket in inputBuckets)
            {
                _rootSearchGraph.AddEdge(new(onode, inputBucket.Vertices.First(), 0, inputBucket));
            }
        }

        return reshardGraph;
    }

    private void Dump(Stream stream, IReadOnlyDictionary<SearchableNode, bool> pickMemo)
    {
        using var writer = new StreamWriter(stream);
        writer.Write(_rootSearchGraph.ToGraphviz(alg =>
        {
            alg.FormatCluster += (_, arg) =>
            {
                if (arg.Cluster is DistributedSearchGraph tg)
                {
                    arg.GraphFormat.LabelLocation = QuikGraph.Graphviz.Dot.GraphvizLabelLocation.T;
                    arg.GraphFormat.LabelJustification = QuikGraph.Graphviz.Dot.GraphvizLabelJustification.L;
                    arg.GraphFormat.Label = tg.Kind.ToString();
                    if (tg.Kind is SearchGraphKind.Bucket)
                    {
                        arg.GraphFormat.Label += ": " + tg.Vertices.First().IRType.ToString();
                    }
                }
            };

            alg.FormatVertex += (_, arg) =>
            {
                var row0 = new QuikGraph.Graphviz.Dot.GraphvizRecordCell();
                var col1 = new QuikGraph.Graphviz.Dot.GraphvizRecordCell();
                row0.Cells.Add(col1);

                col1.Cells.Add(new() { Text = CompilerServices.Print(arg.Vertex.Expr) });
                if (arg.Vertex.Expr is IR.Tuple && arg.Vertex.IRType is TupleType tpTuple)
                {
                    for (int i = 0; i < tpTuple.Fields.Count; i++)
                    {
                        col1.Cells.Add(new() { Text = i.ToString(), Port = $"P{i}" });
                    }
                }
                else if (arg.Vertex.Expr is Op op)
                {
                    for (int i = 0; i < op.Parameters.Count(); i++)
                    {
                        col1.Cells.Add(new() { Text = i.ToString(), Port = $"P{i}" });
                    }
                }

                arg.VertexFormat.Record.Cells.Add(row0);
                arg.VertexFormat.Shape = QuikGraph.Graphviz.Dot.GraphvizVertexShape.Record;
                arg.VertexFormat.Style = QuikGraph.Graphviz.Dot.GraphvizVertexStyle.Filled;
                if (pickMemo.TryGetValue(arg.Vertex, out var picked) && picked == true)
                {
                    arg.VertexFormat.FillColor = QuikGraph.Graphviz.Dot.GraphvizColor.SkyBlue;
                }
            };

            alg.FormatEdge += (_, arg) =>
            {
                arg.EdgeFormat.Direction = QuikGraph.Graphviz.Dot.GraphvizEdgeDirection.Back;
                arg.EdgeFormat.TailPort = $"P{arg.Edge.SourceIndex}";
            };
        }));
    }

    private Expr SolveAndExtract(DistributedSearchGraph rootCluster)
    {
        // 0. create bool var for all node.
        var cpmodel = new CpModel();
        var varMemo = new Dictionary<SearchableNode, BoolVar>();
        var costMemo = new Dictionary<SearchableNode, CostModel.Cost>();
        foreach (var cluster in _rootSearchGraph.Clusters.OfType<DistributedSearchGraph>())
        {
            foreach (var bucket in cluster.Clusters.OfType<DistributedSearchGraph>())
            {
                foreach (var enode in bucket.Vertices)
                {
                    CostModel.Cost cost;
                    switch (enode.Expr)
                    {
                        case Const or Var or If:
                            cost = new CostModel.Cost() { [CostModel.CostFactorNames.CPUCycles] = 1 };
                            break;
                        case Op op:
                            {
                                if (!_rootSearchGraph.TryGetOutEdges(enode, out var edges))
                                {
                                    throw new NotSupportedException("node have no out edges.");
                                }

                                var tempArgs = edges.OrderBy(e => e.SourceIndex).Select<CrossEdge, Expr>(e => e.Target switch
                                {
                                    SearchableNode { Expr: Const attr } => attr,
                                    SearchableNode n => new Var(n.IRType),
                                }).ToArray();

                                var context = new DistributedCostEvaluateContext(op, enode.IRType, tempArgs, CompileOptions);
                                cost = CompilerServices.EvaluateOpCost(op, context);
                            }

                            break;
                        default:
                            throw new NotSupportedException("not");
                    }

                    costMemo.Add(enode, cost);

                    var boolVar = cpmodel.NewBoolVar(string.Empty);
                    varMemo.Add(enode, boolVar);
                }
            }
        }

        // 1. must pick one in root enode.
        cpmodel.AddBoolOr(rootCluster.Vertices.Select(n => varMemo[n]).ToArray());

        // 2. when pick node, must pick one child node.
        foreach (var n in _rootSearchGraph.Vertices)
        {
            var ns = new[] { varMemo[n].Not() };

            if (_rootSearchGraph.TryGetOutEdges(n, out var allEdges))
            {
                foreach (var argEdges in allEdges.GroupBy(g => g.SourceIndex))
                {
                    cpmodel.AddBoolOr(ns.Concat(argEdges.SelectMany(e => e.TargetGraph.Vertices).Select(cn => varMemo[cn])));
                }
            }
        }

        // 3. add pick weights for all enode.
        cpmodel.Minimize(LinearExpr.WeightedSum(_rootSearchGraph.Vertices.Select(n => varMemo[n]), _rootSearchGraph.Vertices.Select(n => checked((long)costMemo[n].Score))));

        if (cpmodel.Validate().Any())
        {
            throw new InvalidDataException("the sat model invalid: " + cpmodel.Validate());
        }

        var solver = new CpSolver();
        int max_time = 120;
        if (System.Environment.GetEnvironmentVariable("SOLVE_MAX_TIME") is string s_solve_max_time)
        {
            try
            {
                var solve_max_time = int.Parse(s_solve_max_time);
                max_time = solve_max_time;
            }
            catch (System.Exception)
            {
            }
        }

        int processorCount = Math.Max(System.Environment.ProcessorCount / 2, 1);
        if (System.Environment.GetEnvironmentVariable("SOLVE_PROCESSOR_COUNT") is string s_solve_processor_count)
        {
            try
            {
                var solve_processor_count = int.Parse(s_solve_processor_count);
                processorCount = solve_processor_count;
            }
            catch (System.Exception)
            {
            }
        }

        solver.StringParameters = $"max_time_in_seconds:{max_time},num_workers:{processorCount}";

        var enableDump = Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.EGraphCost);
        CpSolverStatus status;
        using (var dumpStream = enableDump ? Diagnostics.DumpScope.Current.OpenFile("Costs/Solve.txt") : Stream.Null)
        {
            using var writer = new StreamWriter(dumpStream);
            var cb = new PrintCostCallBack(varMemo, costMemo, writer, enableDump);
            status = solver.Solve(cpmodel, cb);
            writer.WriteLine($"Status : {status}");
        }

        if (status is not (CpSolverStatus.Optimal or CpSolverStatus.Feasible))
        {
            throw new InvalidProgramException("SatExtract Failed!");
        }

        var picks = _rootSearchGraph.Vertices.ToDictionary(e => e, e => solver.BooleanValue(varMemo[e]));
        using (var stream = enableDump ? Diagnostics.DumpScope.Current.OpenFile("Costs/Pick.dot") : Stream.Null)
        {
            Dump(stream, picks);
        }

        return new ExprBuildVisitor(_rootSearchGraph, picks).Visit(rootCluster.Clusters.OfType<DistributedSearchGraph>());
    }
}

internal sealed class ExprBuildVisitor
{
    private readonly Dictionary<SearchableNode, bool> _picks;
    private readonly DistributedSearchGraph _rootSearchGraph;
    private readonly Dictionary<SearchableNode, Expr> _memo;

    public ExprBuildVisitor(DistributedSearchGraph rootSearchGraph, Dictionary<SearchableNode, bool> picks)
    {
        _rootSearchGraph = rootSearchGraph;
        _picks = picks;
        _memo = new();
    }

    public Expr Visit(IEnumerable<DistributedSearchGraph> rootBuckets)
    {
        var rootPicks = rootBuckets.SelectMany(b => b.Vertices).Where(v => _picks.TryGetValue(v, out var pick) && pick).ToArray();
        if (rootPicks.Length != 1)
        {
            throw new InvalidProgramException("the one cluster only can pick one vertex!");
        }

        var root = rootPicks[0];
        if (!_memo.TryGetValue(root, out var expr))
        {
            _rootSearchGraph.TryGetOutEdges(root, out var edges);
            var children = edges.GroupBy(e => e.SourceIndex).Select(g => Visit(g.Select(e => e.TargetGraph))).ToArray();
            switch (root.Expr)
            {
                case Var or TensorConst or TupleConst or Fusion or None:
                    expr = root.Expr;
                    break;
                case Function func:
                    expr = children.Length == 0 ? func : func.With(body: children[0], parameters: children[1..].OfType<Var>().ToArray());
                    break;
                case Op op:
                    expr = new Call(target: op, arguments: children);
                    break;
                case IR.Tuple tp:
                    expr = tp.With(fields: children);
                    break;
                case IR.If @if:
                    expr = @if.With(condition: children[^3], then: children[^2], @else: children[^1], paramList: children[..^3].ToArray());
                    break;
                default:
                    throw new NotSupportedException(root.Expr.GetType().Name);
            }

            _memo.Add(root, expr);
        }

        return expr;
    }
}

internal sealed class DistributedCostEvaluateContext : Evaluator.ICostEvaluateContext
{
    public DistributedCostEvaluateContext(Op op, IRType returnType, Expr[] args, CompileOptions compileOptions)
    {
        Op = op;
        ReturnType = returnType;
        Args = args;
        CompileOptions = compileOptions;
    }

    public Op Op { get; }

    public IRType ReturnType { get; }

    public Expr[] Args { get; }

    public CompileOptions CompileOptions { get; }

    public T GetArgument<T>(Op op, ParameterInfo parameter)
        where T : BaseFunction
    {
        throw new NotSupportedException();
    }

    public T GetArgumentType<T>(Op op, ParameterInfo parameter)
        where T : IRType
    {
        if (op.GetType() == parameter.OwnerType)
        {
            return (T?)Args[parameter.Index].CheckedType ?? throw new InvalidOperationException("Run type infer first.");
        }
        else
        {
            throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
        }
    }

    public T GetReturnType<T>()
         where T : IRType
    {
        return (T)ReturnType;
    }
}

internal sealed class PrintCostCallBack : CpSolverSolutionCallback
{
    private readonly IReadOnlyDictionary<SearchableNode, BoolVar> _vars;
    private readonly Dictionary<SearchableNode, CostModel.Cost> _costModel;
    private readonly StreamWriter _dumpWriter;
    private readonly bool _enableDump;
    private int _count;

    public PrintCostCallBack(IReadOnlyDictionary<SearchableNode, BoolVar> vars, Dictionary<SearchableNode, CostModel.Cost> costModel, StreamWriter writer, bool enableDump)
    {
        _vars = vars;
        _costModel = costModel;
        _dumpWriter = writer;
        _enableDump = enableDump;
    }

    public override void OnSolutionCallback()
    {
        if (_enableDump)
        {
            var cost = CostModel.Cost.Zero;
            foreach (var (n, v) in _vars)
            {
                if (_costModel[n] != CostModel.Cost.Zero && BooleanValue(v))
                {
                    cost += _costModel[n];
                }
            }

            _dumpWriter.WriteLine($"Solution {_count++} @ {WallTime()}:");
            _dumpWriter.WriteLine(cost.ToString());
            _dumpWriter.Flush();
        }
    }
}
