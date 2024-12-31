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

public sealed class AutoDistributedMetadata : IRMetadata
{
    public AutoDistributedMetadata(bool skip)
    {
        Skip = skip;
    }

    /// <summary>
    /// Gets a value indicating whether skip auto distributed.
    /// </summary>
    public bool Skip { get; }
}

/// <summary>
/// auto distributed the function.
/// </summary>
[RuleGenerator]
public sealed partial class AutoDistributedPass : FunctionPass
{
    private readonly CompileOptions _compileOptions;

    private readonly string _moduleKind;

    public AutoDistributedPass(bool bidirectional, CompileOptions compileOptions, string moduleKind = "cpu")
    {
        Bidirectional = bidirectional;
        _compileOptions = compileOptions;
        _moduleKind = moduleKind;
    }

    public bool Bidirectional { get; }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input.Metadata is AutoDistributedMetadata { Skip: true })
        {
            return Task.FromResult(input);
        }

        var rewriter = new AutoDistributedRewriter(Bidirectional, _compileOptions, _compileOptions.TargetOptions is CpuTargetOptions options ? options : new CpuTargetOptions(), _moduleKind);

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

internal sealed record CrossEdge : IEdge<SearchableNode>
{
    public CrossEdge(SearchableNode root, SearchableNode input, int inputIndex, DistributedSearchGraph inputGraph)
    {
        Root = root;
        Input = input;
        InputIndex = inputIndex;
        InputGraph = inputGraph;
    }

    public SearchableNode Root { get; }

    public SearchableNode Input { get; }

    public int InputIndex { get; }

    public DistributedSearchGraph InputGraph { get; }

    public SearchableNode Source => Root;

    public SearchableNode Target => Input;
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

    private readonly string _moduleKind;

    public AutoDistributedRewriter(bool bidirectional, CompileOptions compileOptions, CpuTargetOptions targetOptions, string moduleKind)
    {
        Placements = targetOptions.Hierarchies.Select(h => new Placement(h, targetOptions.HierarchyNames)).ToArray();
        Bidirectional = bidirectional;
        CompileOptions = compileOptions;
        TargetOptions = targetOptions;
        _moduleKind = moduleKind;
        if (Path.Exists(TargetOptions.DistributedScheme) && System.Text.Json.JsonSerializer.Deserialize<DistributedSchema>(File.ReadAllText(TargetOptions.DistributedScheme)) is DistributedSchema scheme)
        {
            Scheme = scheme.Outputs.ToDictionary(n => n.Name, n => (new IRArray<SBP>(n.NdSBP), new Placement(n.Hierarchy, n.HierarchyName, targetOptions.HierarchyKind)));
        }
        else
        {
            Scheme = new Dictionary<string, (IRArray<SBP> NdSBP, Placement Placement)>();
        }

        _reshardMemo = new(ReferenceEqualityComparer.Instance);
        _inferedMemo = new(ReferenceEqualityComparer.Instance);
        _rootGraph = new(true);
        _rootSearchGraph = new(_rootGraph, SearchGraphKind.Root);
    }

    public IRArray<Placement> Placements { get; }

    public bool Bidirectional { get; }

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

                if (k.Expr is Call { Target: IR.CPU.Boxing boxing } call && boxing.NewType is DistributedType distributedType && call.Arguments[0].CheckedType is DistributedType inType && inType.NdSBP.Any(sbp => sbp is SBPPartial) && distributedType != call.Arguments[0].CheckedType)
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
        bool Matched(SearchableNode node, (IRArray<SBP> NdSBP, Placement Placement) tp)
        {
            return node.IRType is DistributedType dtype && dtype.NdSBP == tp.NdSBP && dtype.Placement == tp.Placement;
        }

        foreach (var name in expr.Metadata.OutputNames ?? Array.Empty<string>())
        {
            if (Scheme.TryGetValue(name, out var tp))
            {
                if (cluster.Kind is SearchGraphKind.DistributedCluster)
                {
                    if (!cluster.Clusters.OfType<DistributedSearchGraph>().Any(b => Matched(b.Vertices.First(), tp)))
                    {
                        return;
                    }

                    var removes = new List<DistributedSearchGraph>();
                    foreach (var bucket in cluster.Clusters.OfType<DistributedSearchGraph>())
                    {
                        bucket.RemoveVertexIf(v => !Matched(v, tp));
                        if (bucket.VertexCount == 0)
                        {
                            removes.Add(bucket);
                        }
                    }

                    foreach (var r in removes)
                    {
                        cluster.RemoveCluster(r);
                    }

                    foreach (var bucket in cluster.Clusters.OfType<DistributedSearchGraph>().Where(b => Matched(b.Vertices.First(), tp)))
                    {
                        bucket.RemoveVertexIf(v => _rootSearchGraph.TryGetOutEdges(v, out var edges) && !edges.Any());
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
                Dump(stream, new Dictionary<SearchableNode, bool>() { }, new Dictionary<SearchableNode, CostModel.Cost>() { });
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
        bool isSupported;
        var argClusters = new DistributedSearchGraph[expr.Arguments.Length];
        if (expr.Target is not Op op)
        {
            isSupported = false;
            foreach (var (param, i) in expr.Arguments.AsValueEnumerable().Select((p, i) => (p, i)))
            {
                argClusters[i] = VisitLeafArgument(ParameterKind.Input, expr.Arguments[i], isSupported);
            }
        }
        else
        {
            isSupported = PassUtility.IsCpuSupported(op, expr, expr.Arguments.ToArray(), _moduleKind);
            foreach (var param in op.Parameters)
            {
                argClusters[param.Index] = VisitLeafArgument(param.ParameterKind, expr.Arguments[param.Index], isSupported);
            }
        }

        var callCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(isSupported ? SearchGraphKind.DistributedCluster : SearchGraphKind.StandaloneCluster);

        // 1. inference
        var bucketMemo = new Dictionary<IRType, DistributedSearchGraph>();
        foreach (var combBuckets in argClusters.Select(c => c.Clusters.OfType<DistributedSearchGraph>()).CartesianProduct())
        {
            var tempArgs = combBuckets.Select<DistributedSearchGraph, Expr>(bucket => bucket.Vertices.First() switch
            {
                SearchableNode { Expr: Const attr } => attr,
                SearchableNode n => new Var(n.IRType),
            }).ToArray();
            var newCalls = BuildEquivalentCalls(expr.Target, tempArgs);
            foreach (var (newCall, used) in newCalls)
            {
                var checkType = newCall.CheckedType;
                if (checkType is InvalidType)
                {
                    continue;
                }

                if (!bucketMemo.TryGetValue(checkType, out var dbucket))
                {
                    dbucket = callCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                    bucketMemo.Add(checkType, dbucket);
                }

                var dnode = new SearchableNode(newCall.Target, checkType);
                dbucket.AddVertex(dnode);

                foreach (var ((arg, _), i) in combBuckets.Zip(used).Where(p => p.Second is true).Select((arg, i) => (arg, i)))
                {
                    _rootSearchGraph.AddEdge(new(dnode, arg.Vertices.First(), i, arg));
                }
            }
        }

        _inferedMemo.Add(expr, callCluster);

        if (!isSupported)
        {
            return default;
        }

        // 3. add bidirectional connections.
        if (Bidirectional)
        {
            foreach (var (lType, lBucket) in bucketMemo.Where(kv => kv.Key is DistributedType))
            {
                foreach (var (rType, rBucket) in bucketMemo.Where(kv => kv.Key is DistributedType distributedType && distributedType != lType))
                {
                    if (Evaluator.IR.CPU.BoxingEvaluator.VisitType(lType, rType) is not InvalidType)
                    {
                        var rnode = new SearchableNode(new Boxing(rType), rType);
                        rBucket.AddVertex(rnode);
                        callCluster.AddEdge(new(rnode, lBucket.Vertices.First(), 0, lBucket));
                    }
                }
            }
        }

        // 4. add not infered type in search space.
        var addedBuckets = bucketMemo.Values.ToArray();
        foreach (var nType in GetLeafCandidateDistTypes(expr.CheckedTensorType, Placements))
        {
            if (!bucketMemo.TryGetValue(nType, out var bucket))
            {
                bucket = callCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var node = new SearchableNode(new Boxing(nType), nType);
                bucket.AddVertex(node);

                var linked = false;
                foreach (var addedBucket in addedBuckets)
                {
                    var addedNode = addedBucket.Vertices.First();
                    if (Evaluator.IR.CPU.BoxingEvaluator.VisitType(addedNode.IRType, nType) is not InvalidType)
                    {
                        callCluster.AddEdge(new(node, addedNode, 0, addedBucket));
                        linked |= true;
                    }
                }

                if (!linked)
                {
                    bucket.RemoveVertex(node);
                    callCluster.RemoveCluster(bucket);
                }
                else
                {
                    bucketMemo.Add(nType, bucket);
                }
            }
        }

        // 5. filter
        FilterByScheme(expr, callCluster);
        return default;
    }

    private IEnumerable<(Call Call, bool[] Used)> BuildEquivalentCalls(Expr target, Expr[] tempArgs)
    {
        IEnumerable<(Call Call, bool[] Used)> calls = [(new Call(target, tempArgs), Enumerable.Repeat(true, tempArgs.Length).ToArray())];
        if (target is IR.Tensors.Reshape reshape && tempArgs[0].CheckedType is DistributedType distType && tempArgs[1] is TensorConst tc)
        {
            var newShape = tc.Value.ToArray<int>();
            var newTensorType = new TensorType(distType.TensorType.DType, newShape);
            calls = calls.Concat(DistributedUtility.GetLeafCandidateNDSBPs(newTensorType, distType.Placement).Select(ndsbp => (new Call(new Boxing(new DistributedType(newTensorType, ndsbp, distType.Placement)), tempArgs[0]), new[] { true, false })));
        }

        return calls;
    }

    private DistributedSearchGraph VisitLeafArgument(ParameterKind parameterKind, Expr expr, bool isSupported)
    {
        DistributedSearchGraph argCluster;
        switch (parameterKind, expr)
        {
            case (ParameterKind.Input, Expr e):
                if (isSupported)
                {
                    argCluster = TryAddOriginator(e);
                }
                else
                {
                    argCluster = TryInstertTerminator(e);
                }

                break;
            case (ParameterKind.Attribute, Expr e):
                argCluster = TryInstertTerminator(e);
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

    private DistributedSearchGraph CreateOriginatorCluster(Expr expr, bool init)
    {
        if (expr is IR.Tuple tp)
        {
            var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
            var buckets = new DistributedSearchGraph[tp.Fields.Length];
            foreach (var (f, fGraph, i) in tp.Fields.AsValueEnumerable().Select((f, i) => (f, Visit(f), i)))
            {
                buckets[i] = TryAddOriginator(f).Clusters.OfType<DistributedSearchGraph>().First();
            }

            var tpnode = new SearchableNode(new IR.Tuple(), new TupleType(buckets.Select(g => g.Vertices.First().IRType).ToArray()));
            var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            bucket.AddVertex(tpnode);
            for (int i = 0; i < tp.Fields.Length; i++)
            {
                _rootSearchGraph.AddEdge(new(tpnode, buckets[i].Vertices.First(), i, buckets[i]));
            }

            return distCluster;
        }
        else if (expr is TensorConst tc && tc.ValueType is TensorType tensorType)
        {
            var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
            foreach (var dType in GetLeafCandidateDistTypes(tensorType, Placements))
            {
                var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var dnode = new SearchableNode(new TensorConst(tc.Value, dType.NdSBP, dType.Placement), dType);
                bucket.AddVertex(dnode);
            }

            return distCluster;
        }
        else
        {
            if (init)
            {
                var standCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.StandaloneCluster);
                var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var node = new SearchableNode(expr, expr.CheckedType);
                bucket.AddVertex(node);
                return standCluster;
            }
            else
            {
                var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
                var inferCluster = _inferedMemo[expr];
                foreach (var dType in GetLeafCandidateDistTypes((TensorType)inferCluster.Vertices.First().IRType, Placements))
                {
                    var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                    var dnode = new SearchableNode(new Boxing(dType), dType);
                    bucket.AddVertex(dnode);
                    _rootSearchGraph.AddEdge(new(dnode, inferCluster.Vertices.First(), 0, inferCluster.Clusters.OfType<DistributedSearchGraph>().First()));
                }

                return distCluster;
            }
        }
    }

    private DistributedSearchGraph TryAddOriginator(Expr expr)
    {
        if (!_inferedMemo.TryGetValue(expr, out var inferCluster))
        {
            inferCluster = CreateOriginatorCluster(expr, true);
            _inferedMemo.Add(expr, inferCluster);
        }

        if (inferCluster.Kind is SearchGraphKind.DistributedCluster)
        {
            return inferCluster;
        }

        // unshard to standalone
        if (!_reshardMemo.TryGetValue(expr, out var distCluster))
        {
            distCluster = CreateOriginatorCluster(expr, false);
            _reshardMemo.Add(expr, distCluster);
        }

        if (distCluster.Kind != SearchGraphKind.DistributedCluster)
        {
            throw new InvalidOperationException("The inference and reshard cluster cannot be distributed either.");
        }

        return distCluster;
    }

    private DistributedSearchGraph CreateTerminatorCluster(Expr expr, bool init)
    {
        var standCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.StandaloneCluster);

        if (expr is IR.Tuple tp)
        {
            var buckets = new DistributedSearchGraph[tp.Fields.Length];
            foreach (var (f, fGraph, i) in tp.Fields.AsValueEnumerable().Select((f, i) => (f, Visit(f), i)))
            {
                buckets[i] = TryInstertTerminator(f).Clusters.OfType<DistributedSearchGraph>().First();
            }

            var tpnode = new SearchableNode(new IR.Tuple(), new TupleType(buckets.Select(g => g.Vertices.First().IRType).ToArray()));
            var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            bucket.AddVertex(tpnode);
            for (int i = 0; i < tp.Fields.Length; i++)
            {
                _rootSearchGraph.AddEdge(new(tpnode, buckets[i].Vertices.First(), i, buckets[i]));
            }
        }
        else
        {
            if (init)
            {
                var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var node = new SearchableNode(expr, expr.CheckedType);
                bucket.AddVertex(node);
            }
            else
            {
                var onode = new SearchableNode(new Boxing(expr.CheckedType), expr.CheckedType);
                var inputBuckets = _inferedMemo[expr].Clusters.OfType<DistributedSearchGraph>().ToArray();

                var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                bucket.AddVertex(onode);
                foreach (var inputBucket in inputBuckets)
                {
                    if (Evaluator.IR.CPU.BoxingEvaluator.VisitType(inputBucket.Vertices.First().IRType, onode.IRType) is not InvalidType)
                    {
                        _rootSearchGraph.AddEdge(new(onode, inputBucket.Vertices.First(), 0, inputBucket));
                    }
                }
            }
        }

        return standCluster;
    }

    private DistributedSearchGraph TryInstertTerminator(Expr expr)
    {
        if (!_inferedMemo.TryGetValue(expr, out var inferCluster))
        {
            inferCluster = CreateTerminatorCluster(expr, true);
            _inferedMemo.Add(expr, inferCluster);
            return inferCluster;
        }

        if (inferCluster.Kind is SearchGraphKind.StandaloneCluster)
        {
            return inferCluster;
        }

        // unshard to standalone
        if (!_reshardMemo.TryGetValue(expr, out var standCluster))
        {
            standCluster = CreateTerminatorCluster(expr, false);
            _reshardMemo.Add(expr, standCluster);
            return standCluster;
        }

        if (standCluster.Kind != SearchGraphKind.StandaloneCluster)
        {
            throw new InvalidOperationException("The inference and reshard cluster cannot be distributed either.");
        }

        return standCluster;
    }

    private void Dump(Stream stream, IReadOnlyDictionary<SearchableNode, bool> pickMemo, IReadOnlyDictionary<SearchableNode, CostModel.Cost> costMemo)
    {
        using var writer = new StreamWriter(stream);
        writer.Write(_rootSearchGraph.ToGraphviz(alg =>
        {
            alg.GraphFormat.RankDirection = QuikGraph.Graphviz.Dot.GraphvizRankDirection.LR;
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
                if (costMemo.TryGetValue(arg.Vertex, out var cost))
                {
                    var row1 = new QuikGraph.Graphviz.Dot.GraphvizRecordCell();
                    foreach (var (k, v) in cost.Factors)
                    {
                        row1.Cells.Add(new() { Text = $"{k}: {v}" });
                    }

                    row1.Cells.Add(new() { Text = $"Score: {cost.Score}" });
                    col1.Cells.Add(row1);
                }

                if (pickMemo.TryGetValue(arg.Vertex, out var picked) && picked == true)
                {
                    arg.VertexFormat.FillColor = QuikGraph.Graphviz.Dot.GraphvizColor.SkyBlue;
                }
            };

            alg.FormatEdge += (_, arg) =>
            {
                arg.EdgeFormat.Direction = QuikGraph.Graphviz.Dot.GraphvizEdgeDirection.Back;
                arg.EdgeFormat.TailPort = $"P{arg.Edge.InputIndex}";
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
                        case Const or Var or If or IR.Tuple or BaseFunction:
                            cost = new CostModel.Cost() { [CostModel.CostFactorNames.CPUCycles] = 1 };
                            break;
                        case Op op:
                            {
                                if (!_rootSearchGraph.TryGetOutEdges(enode, out var edges))
                                {
                                    throw new NotSupportedException("graph doesn't contain the vertex.");
                                }

                                var tempArgs = edges.OrderBy(e => e.InputIndex).Select<CrossEdge, Expr>(e => e.Target switch
                                {
                                    SearchableNode { Expr: Const attr } => attr,
                                    SearchableNode n => new Var(n.IRType),
                                }).ToArray();

                                var context = new DistributedCostEvaluateContext(op, enode.IRType, tempArgs, CompileOptions);
                                cost = CompilerServices.EvaluateOpCost(op, context);
                            }

                            break;
                        default:
                            throw new NotSupportedException($"extract not support {enode.Expr.GetType()}");
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
                foreach (var argEdges in allEdges.GroupBy(g => g.InputIndex))
                {
                    cpmodel.AddBoolOr(ns.Concat(argEdges.SelectMany(e => e.InputGraph.Vertices).Select(cn => varMemo[cn])));
                }
            }
        }

        // 3. no cycle
        {
            var hgraph = ToHyperGraph(_rootSearchGraph, rootCluster);
            var class_cycles = hgraph.FindCycles();
            foreach (var cycle in class_cycles)
            {
                if (cycle.Count == 1)
                {
                    foreach (var n in cycle[0].Vertices)
                    {
                        _rootSearchGraph.TryGetOutEdges(n, out var edgs);
                        if (edgs.Select(e => e.InputGraph).Contains(cycle[0]))
                        {
                            cpmodel.AddAssumption(varMemo[n].Not());
                        }
                    }
                }
                else
                {
                    // build clauses.
                    var clauses = new List<List<BoolVar>>();
                    for (int i = 0; i < cycle.Count; i++)
                    {
                        var next_hop = (i + 1) % cycle.Count;
                        var u = hgraph.Edges(cycle[i])!;
                        var v = u[cycle[next_hop]];
                        clauses.Add(v.Select(n => varMemo[n]).ToList());
                    }

                    var clauseMemo = new Dictionary<int, BoolVar>();
                    for (int i = 0; i < clauses.Count; i++)
                    {
                        var clause = clauses[i];
                        if (clause.Count > 1)
                        {
                            var tmpV = cpmodel.NewBoolVar(string.Empty);
                            cpmodel.AddBoolAnd(clause.Select(c => c.Not())).OnlyEnforceIf(tmpV);
                            cpmodel.AddBoolOr(clause).OnlyEnforceIf(tmpV.Not());
                            clauseMemo.Add(i, tmpV);
                        }
                    }

                    cpmodel.AddBoolOr(clauses.Select((c, i) => (c, i)).Select(p => p.c.Count == 1 ? p.c[0].Not() : clauseMemo[p.i]));
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
            Dump(stream, picks, costMemo);
        }

        return new ExprBuildVisitor(_rootSearchGraph, picks).Visit(rootCluster.Clusters.OfType<DistributedSearchGraph>());
    }

    private HyperGraph<DistributedSearchGraph, SearchableNode> ToHyperGraph(DistributedSearchGraph root, DistributedSearchGraph rootCluster)
    {
        var hgraph = new HyperGraph<DistributedSearchGraph, SearchableNode>();
        var visited = new HashSet<DistributedSearchGraph>();
        var queue = new Queue<DistributedSearchGraph>();
        var rootBuckets = rootCluster.Clusters.OfType<DistributedSearchGraph>().ToArray();
        if (rootBuckets.Length != 1)
        {
            throw new InvalidOperationException("The root Cluster should contains only one bucket!");
        }

        queue.Enqueue(rootBuckets[0]);
        visited.Add(rootBuckets[0]);
        while (queue.Any())
        {
            var front = queue.Dequeue();
            foreach (var node in front.Vertices)
            {
                root.TryGetOutEdges(node, out var edges);
                foreach (var edge in edges)
                {
                    var canonical = edge.InputGraph;
                    hgraph.Connect(front, canonical, node);
                    if (!visited.Contains(canonical))
                    {
                        visited.Add(canonical);
                        queue.Enqueue(canonical);
                    }
                }
            }
        }

        return hgraph;
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
            var children = edges.GroupBy(e => e.InputIndex).Select(g => Visit(g.Select(e => e.InputGraph))).ToArray();
            switch (root.Expr)
            {
                case Var or TensorConst or TupleConst or None:
                    expr = root.Expr;
                    break;
                case BaseFunction func:
                    expr = new Call(target: func, arguments: children);
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
