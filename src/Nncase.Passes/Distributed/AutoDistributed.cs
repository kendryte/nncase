// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Google.OrTools.Sat;
using NetFabric.Hyperlinq;
using Nncase.Evaluator;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.NN;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Targets;
using Nncase.Utilities;
using QuikGraph;
using QuikGraph.Graphviz;

[assembly: InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.Passes.Distributed;

public enum AutoDistributedPhase
{
    SearchConstant,
    Final,
}

internal enum SearchGraphKind : int
{
    Root,
    DistributedCluster,
    StandaloneCluster,
    Bucket,
}

public sealed class AutoDistributedMetaData : IRMetadata
{
    public bool Skip { get; set; }
}

/// <summary>
/// auto distributed the function.
/// </summary>
public sealed partial class AutoDistributedPass : FunctionPass
{
    private readonly CompileOptions _compileOptions;

    private readonly bool _bidirectional;

    private readonly string _moduleKind;

    public AutoDistributedPass(bool bidirectional, string moduleKind, CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;
        _bidirectional = bidirectional;
        _moduleKind = moduleKind;
    }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function function || input.Metadata is AutoDistributedMetaData { Skip: true })
        {
            return Task.FromResult(input);
        }

        if (_compileOptions.TargetOptions is INTTTargetOptions targetOptions)
        {
            var rewriter = new AutoDistributedRewriter(_compileOptions, targetOptions, AutoDistributedPhase.Final, _moduleKind, _bidirectional);
            return Task.FromResult((BaseFunction)rewriter.Rewrite(function));
        }

        return Task.FromResult(input);
    }
}

internal sealed class SearchableNode
{
    public SearchableNode(BaseExpr expr, IRType type, bool isBidirect = false)
    {
        Expr = expr;
        IRType = type;
        IsBidirect = isBidirect;
    }

    public BaseExpr Expr { get; }

    public IRType IRType { get; }

    public bool IsBidirect { get; }
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
    private readonly Dictionary<BaseExpr, DistributedSearchGraph> _reshardMemo;

    private readonly Dictionary<BaseExpr, DistributedSearchGraph> _inferedMemo;

    private readonly AdjacencyGraph<SearchableNode, CrossEdge> _rootGraph;

    private readonly DistributedSearchGraph _rootSearchGraph;

    private readonly string _moduleKind;

    private readonly bool _bidirectional;

    private readonly AutoDistributedPhase _phase;

    private readonly Dictionary<Type, ITypeInferencer> _inferencer_cache = new Dictionary<Type, ITypeInferencer>();

    /// <summary>
    /// The original tensor consts that are distributed.
    /// </summary>
    private readonly Dictionary<TensorConst, TensorConst> _distributedConstSources = new(ReferenceEqualityComparer.Instance);

    public AutoDistributedRewriter(CompileOptions compileOptions, INTTTargetOptions targetOptions, AutoDistributedPhase phase, string moduleKind = "cpu", bool bidirectional = false)
    {
        Placements = targetOptions.Hierarchies.Select(h => new Placement(h, targetOptions.HierarchyNames, targetOptions.HierarchyKind)).ToArray();
        Bidirectional = bidirectional;
        CompileOptions = compileOptions;
        TargetOptions = targetOptions;
        _moduleKind = moduleKind;
        _phase = phase;
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
        _moduleKind = moduleKind;
        _bidirectional = bidirectional;
    }

    public IRArray<Placement> Placements { get; }

    public bool Bidirectional { get; }

    public CompileOptions CompileOptions { get; }

    public INTTTargetOptions TargetOptions { get; }

    public IReadOnlyDictionary<string, (IRArray<SBP> Policies, Placement Placement)> Scheme { get; }

    /// <summary>
    /// Gets the final distributed consts that are used in the function.
    /// </summary>
    public Dictionary<TensorConst, TensorConst> DistributedConsts { get; } = new(ReferenceEqualityComparer.Instance);

    public static void MemoryExtractConstrains(CpModel model, IReadOnlyDictionary<ENode, BoolVar> vars)
    {
        var consts = vars.Keys.Where(k => k.Expr is Call { Target: IR.Distributed.Boxing { NewType: DistributedType } } call && call.Arguments[0] is TensorConst tc && tc.Value.Length >= 8).ToArray();
        model.Add(LinearExpr.WeightedSum(consts.Select(k => vars[k]), consts.Select(k =>
        {
            var type = DistributedUtility.GetDividedTensorType((DistributedType)k.Expr.CheckedType);
            var maxShape = CompilerServices.GetMaxShape(type.Shape);
            return TensorUtilities.GetProduct(maxShape) * type.DType.SizeInBytes;
        })) < (2L * 512L * 1024L * 1024L));
    }

    public static bool SingleNodeMemoryCheck(DistributedType distributedType, string moduleKind, INTTTargetOptions targetOptions)
    {
        if (moduleKind == "xpu")
        {
            var type = DistributedUtility.GetDividedTensorType(distributedType);
            var maxShape = CompilerServices.GetMaxShape(type.Shape);
            var size = TensorUtilities.GetProduct(maxShape) * type.DType.SizeInBytes;

            return size < targetOptions.HierarchySizes[^2] / targetOptions.Hierarchies[0][^1];
        }

        return true;
    }

    public static IReadOnlyList<DistributedType> GetLeafCandidateDistTypes(TensorType tensorType, IEnumerable<Placement> placements, string moduleKind, INTTTargetOptions targetOptions)
    {
        return placements.Select(
            placement =>
            DistributedUtility.GetLeafCandidatePolicies(tensorType, placement)
            .Where(p => SingleNodeMemoryCheck(new(tensorType, p, placement), moduleKind, targetOptions))
            .Select(ndsbp => new DistributedType(tensorType, ndsbp, placement)))
            .SelectMany(e => e).ToArray();
    }

    public void SingleNodeMemoryExtractConstrains(CpModel model, IReadOnlyDictionary<ENode, BoolVar> vars)
    {
        var distTypes = vars.Keys.Where(k => k.Expr.CheckedType is DistributedType dt).ToArray();
        foreach (var k in distTypes)
        {
            if (TargetOptions.HierarchySizes.Length > 1)
            {
                var type = DistributedUtility.GetDividedTensorType((DistributedType)k.Expr.CheckedType);
                var maxShape = CompilerServices.GetMaxShape(type.Shape);
                var size = TensorUtilities.GetProduct(maxShape) * type.DType.SizeInBytes;

                if (k.Expr is Call call)
                {
                    for (var i = 0; i < call.Arguments.Length; i++)
                    {
                        if (call.Arguments[i].CheckedType is DistributedType inType)
                        {
                            type = DistributedUtility.GetDividedTensorType(inType);
                            size += TensorUtilities.GetProduct(type.Shape.ToValueArray()) * type.DType.SizeInBytes;
                        }
                    }
                }

                model.Add(vars[k] * size < TargetOptions.HierarchySizes[^2] / TargetOptions.Hierarchies[0][^1]);
            }
        }
    }

    public void FilterByScheme(BaseExpr expr, DistributedSearchGraph cluster)
    {
        bool Matched(SearchableNode node, (IRArray<SBP> Policies, Placement Placement) tp)
        {
            return node.IRType is DistributedType dtype && dtype.AxisPolicies == tp.Policies && dtype.Placement == tp.Placement;
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

    public Function Rewrite(Function function)
    {
        var body = function.Body;
        Visit(body);
        var rootCluster = TryInstertTerminator(body);

#if true
        using (var stream = Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.PassIR) ? Diagnostics.DumpScope.Current.OpenFile("DistributedSearchGraph.dot") : Stream.Null)
        {
            Dump(stream, new Dictionary<SearchableNode, bool>() { }, new Dictionary<SearchableNode, CostModel.Cost>() { });
        }
#endif

        var post = SolveAndExtract(rootCluster);
        return function.With(body: post);
    }

    protected override Unit DefaultVisitLeaf(BaseExpr expr)
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
            isSupported = expr.Target is AsTensor or IR.Tensors.Range ? false : true;
            foreach (var param in op.Parameters)
            {
                argClusters[param.Index] = VisitLeafArgument(param.ParameterKind, expr.Arguments[param.Index], isSupported);
            }
        }

        bool isStandalone = expr.Target is IR.NN.UpdatePagedAttentionKVCache;
        var callCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(!isSupported || isStandalone ? SearchGraphKind.StandaloneCluster : SearchGraphKind.DistributedCluster);

        // 1. inference
        var bucketMemo = new Dictionary<IRType, DistributedSearchGraph>();
        foreach (var combBuckets in argClusters.Select(c => c.Clusters.OfType<DistributedSearchGraph>()).CartesianProduct())
        {
            var tempArgs = combBuckets.Select<DistributedSearchGraph, BaseExpr>(bucket => bucket.Vertices.First() switch
            {
                SearchableNode { Expr: Dimension attr } => attr,
                SearchableNode { Expr: Shape attr } => attr,
                SearchableNode { Expr: Padding attr } => attr,
                SearchableNode { Expr: Paddings attr } => attr,
                SearchableNode { Expr: Const attr } => attr,
                SearchableNode { Expr: Call { Target: AsTensor } attr } => attr,
                SearchableNode n => new Var(n.IRType),
            }).ToArray();
            var newExprs = BuildEquivalentCalls(expr.Target, tempArgs);
            foreach (var (newExpr, used) in newExprs)
            {
                if (!newExpr.InferenceType(_inferencer_cache) || newExpr.CheckedType is InvalidType)
                {
                    continue;
                }

                var checkType = newExpr.CheckedType;
                if (!bucketMemo.TryGetValue(checkType, out var dbucket))
                {
                    dbucket = callCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                    bucketMemo.Add(checkType, dbucket);
                }

                var dnode = new SearchableNode(isSupported && newExpr is Call newCall ? newCall.Target : newExpr, checkType);
                dbucket.AddVertex(dnode);

                foreach (var ((arg, _), i) in combBuckets.Zip(used).Where(p => p.Second is true).Select((arg, i) => (arg, i)))
                {
                    _rootSearchGraph.AddEdge(new(dnode, arg.Vertices.First(), i, arg));
                }
            }
        }

        if (callCluster.VertexCount == 0)
        {
            throw new InvalidOperationException("Please Check expr's TypeInfer.");
        }

        _inferedMemo.Add(expr, callCluster);

        if (!isSupported || isStandalone)
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
                    if (CheckBoxingType(lType, rType) is not InvalidType)
                    {
                        var rnode = new SearchableNode(new Boxing(rType), rType, true);
                        rBucket.AddVertex(rnode);
                        callCluster.AddEdge(new(rnode, lBucket.Vertices.First(), 0, lBucket));
                    }
                }
            }
        }

        // 4. add not infered type in search space.
        var addedBuckets = bucketMemo.Values.ToArray();
        foreach (var nType in GetLeafCandidateDistTypes(expr.CheckedTensorType, Placements, _moduleKind, TargetOptions))
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
                    if (CheckBoxingType(addedNode.IRType, nType) is not InvalidType)
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

    /// <summary>
    /// some times we didn't use all args.
    /// </summary>
    private IEnumerable<(Expr Call, bool[] Used)> BuildEquivalentCalls(Expr target, BaseExpr[] tempArgs)
    {
        IEnumerable<(Expr Call, bool[] Used)> calls = [(new Call(target, tempArgs), Enumerable.Repeat(true, tempArgs.Length).ToArray())];
        if (target is IR.Tensors.Reshape && tempArgs[0].CheckedType is DistributedType distType && tempArgs[1] is Shape { IsFixed: true } constNewShape)
        {
            var newTensorType = new TensorType(distType.TensorType.DType, constNewShape);
            calls = calls.Concat(DistributedUtility.GetLeafCandidatePolicies(newTensorType, distType.Placement)
                .Where(p => SingleNodeMemoryCheck(new(newTensorType, p, distType.Placement), _moduleKind, TargetOptions))
                .Select(ndsbp => ((Expr)new Call(new Boxing(new DistributedType(newTensorType, ndsbp, distType.Placement)), tempArgs[0]), new[] { true, false })));
        }
        else if (target is Boxing { NewType: TensorType } && tempArgs[0] is TensorConst tc && tc.ValueType is DistributedType distributedType)
        {
            calls = [((Expr)tc, new[] { true })];
        }
        else if (target is GetPositionIds)
        {
            var tensorType = (TensorType)calls.First().Call.CheckedType;
            calls = calls.Concat(GetLeafCandidateDistTypes(tensorType, Placements, _moduleKind, TargetOptions)
                .Select(dt => ((Expr)IR.F.NN.GetPositionIds((Dimension)tempArgs[0], (Expr)tempArgs[1], dt.AxisPolicies, dt.Placement), new[] { true, true })));
        }

        return calls;
    }

    private IReadOnlyList<IRArray<SBP>> GetDiverseCandidateSBPs(DistributedType distributedType, IEnumerable<Placement> placements)
    {
        return placements.Select(
            placement =>
                DistributedUtility.GetLeafCandidatePolicies(distributedType.TensorType, placement).
                Where(p => SingleNodeMemoryCheck(new(distributedType.TensorType, p, placement), _moduleKind, TargetOptions)).
                Where(ndsbp => ndsbp != distributedType.AxisPolicies)).
            SelectMany(e => e).ToArray();
    }

    private DistributedSearchGraph VisitLeafArgument(ParameterKind parameterKind, BaseExpr expr, bool isSupported)
    {
        DistributedSearchGraph argCluster;
        switch (parameterKind, expr)
        {
            case (ParameterKind.Input, BaseExpr e):
                if (isSupported)
                {
                    argCluster = TryAddOriginator(e);
                }
                else
                {
                    argCluster = TryInstertTerminator(e);
                }

                break;
            case (ParameterKind.Attribute, BaseExpr e):
                argCluster = TryInstertTerminator(e);
                break;
            case (_, Dimension e):
                argCluster = TryInstertTerminator(e);
                break;
            case (_, Shape e):
                argCluster = TryInstertTerminator(e);
                break;
            case (_, Padding e):
                argCluster = TryInstertTerminator(e);
                break;
            case (_, Paddings e):
                argCluster = TryInstertTerminator(e);
                break;
            case (_, None e):
                argCluster = TryInstertTerminator(e.With());
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

    private DistributedSearchGraph CreateOriginatorCluster(BaseExpr expr, bool init)
    {
        if (expr is IR.Tuple tp)
        {
            var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
            var buckets = new List<DistributedSearchGraph>[tp.Fields.Length];
            foreach (var (f, fGraph, i) in tp.Fields.AsValueEnumerable().Select((f, i) => (f, Visit(f), i)))
            {
                buckets[i] = TryAddOriginator(f).Clusters.OfType<DistributedSearchGraph>().ToList();
            }

            var combBuckets = buckets.CartesianProduct();
            foreach (var comb in combBuckets)
            {
                var tpnode = new SearchableNode(new IR.Tuple(), new TupleType(comb.Select(g => g.Vertices.First().IRType).ToArray()));
                var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                bucket.AddVertex(tpnode);
                for (int i = 0; i < tp.Fields.Length; i++)
                {
                    _rootSearchGraph.AddEdge(new(tpnode, comb.ElementAt(i).Vertices.First(), i, comb.ElementAt(i)));
                }
            }

            return distCluster;
        }
        else if (expr is Call { Target: Boxing { NewType: TensorType } } call && call[Boxing.Input] is TensorConst tc && tc.ValueType is DistributedType distributedType)
        {
            var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
            var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var dnode = new SearchableNode(tc, distributedType);
            bucket.AddVertex(dnode);

            return distCluster;
        }
        else if (expr is TensorConst tc2)
        {
            if (tc2.ValueType is TensorType tensorType)
            {
                var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
                foreach (var dType in GetLeafCandidateDistTypes(tensorType, Placements, _moduleKind, TargetOptions))
                {
                    var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                    var distConst = new TensorConst(tc2.Value, dType.AxisPolicies, dType.Placement);
                    if (_phase == AutoDistributedPhase.SearchConstant)
                    {
                        _distributedConstSources.Add(distConst, tc2);
                    }

                    var dnode = new SearchableNode(distConst, dType);
                    bucket.AddVertex(dnode);
                }

                return distCluster;
            }
            else if (tc2.ValueType is DistributedType distributedType2)
            {
                var distCluster = _rootSearchGraph.CreateCluster<DistributedSearchGraph>(SearchGraphKind.DistributedCluster);
                var bucket = distCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
                var dnode = new SearchableNode(tc2, distributedType2);
                bucket.AddVertex(dnode);

                return distCluster;
            }
            else
            {
                throw new InvalidOperationException($"Unsupported TensorConst type: {tc2.ValueType}");
            }
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
                foreach (var dType in GetLeafCandidateDistTypes((TensorType)inferCluster.Vertices.First().IRType, Placements, _moduleKind, TargetOptions))
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

    private DistributedSearchGraph TryAddOriginator(BaseExpr expr)
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

    private DistributedSearchGraph CreateTerminatorCluster(BaseExpr expr, bool init)
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
        else if (expr is TensorConst tc && tc.ValueType is TensorType tensorType)
        {
            var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var node = new SearchableNode(expr, expr.CheckedType);
            bucket.AddVertex(node);
        }
        else if (expr is Shape or Padding or Paddings or Dimension or None)
        {
            var bucket = standCluster.CreateCluster<DistributedSearchGraph>(SearchGraphKind.Bucket);
            var node = new SearchableNode(expr, expr.CheckedType);
            bucket.AddVertex(node);
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
                    if (CheckBoxingType(inputBucket.Vertices.First().IRType, onode.IRType) is not InvalidType)
                    {
                        _rootSearchGraph.AddEdge(new(onode, inputBucket.Vertices.First(), 0, inputBucket));
                    }
                }
            }
        }

        return standCluster;
    }

    private IRType CheckBoxingType(IRType inType, IRType outType, bool isReshape = false)
    {
        IRType VisitD2D(DistributedType inv, DistributedType outv)
        {
            if (inv == outv)
            {
                return new InvalidType("Same DistributedType");
            }

            if (inv.AxisPolicies.Any(s => s is SBPPartial) || outv.AxisPolicies.Any(s => s is SBPPartial))
            {
                return new InvalidType("Not supported input/output is Partial");
            }

            return outv;
        }

        IRType VisitD2T(DistributedType inv, TensorType outv)
        {
            if (inv.AxisPolicies.Any(s => s is SBPPartial))
            {
                return new InvalidType("Not supported input is Partial output is Unshard");
            }

            return outv;
        }

        IRType VisitT2D(TensorType inv, DistributedType outv)
        {
            if (outv.AxisPolicies.Any(s => s is SBPPartial))
            {
                return new InvalidType("Not supported input is Unshard output is Partial");
            }

            return outv;
        }

        return (inType, outType) switch
        {
            (InvalidType inv, _) => inv,
            (_, InvalidType inv) => inv,
            (DistributedType d, DistributedType d1) => VisitD2D(d, d1),
            (TensorType t, DistributedType d) => VisitT2D(t, d),
            (DistributedType d, TensorType t) => VisitD2T(d, t),
            _ => new InvalidType($"not support boxing {inType} to {outType}"),
        };
    }

    private DistributedSearchGraph TryInstertTerminator(BaseExpr expr)
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
                    if (tg.Kind is SearchGraphKind.Bucket && tg.Vertices.Any())
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

                col1.Cells.Add(new() { Text = arg.Vertex.Expr.GetType().ToString() });
                if (arg.Vertex.Expr is IR.Tuple && arg.Vertex.IRType is TupleType tpTuple)
                {
                    for (int i = 0; i < tpTuple.Fields.Count; i++)
                    {
                        col1.Cells.Add(new() { Text = i.ToString(), Port = $"P{i}" });
                    }
                }
                else if (arg.Vertex.Expr is Op op)
                {
                    for (int i = 0; i < op.Parameters.Count; i++)
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

    private BaseExpr SolveAndExtract(DistributedSearchGraph rootCluster)
    {
        // 0. create bool var for all node.
        var cpmodel = new CpModel();
        var varMemo = new Dictionary<SearchableNode, BoolVar>();
        var clusterVarMemo = new Dictionary<DistributedSearchGraph, List<BoolVar>>();
        var costMemo = new Dictionary<SearchableNode, CostModel.Cost>();
        foreach (var cluster in _rootSearchGraph.Clusters.OfType<DistributedSearchGraph>())
        {
            clusterVarMemo.Add(cluster, new());
            foreach (var bucket in cluster.Clusters.OfType<DistributedSearchGraph>())
            {
                foreach (var enode in bucket.Vertices)
                {
                    CostModel.Cost cost;
                    switch (enode.Expr)
                    {
                        case Const or Var or If or IR.Tuple or BaseFunction or Shape or Padding or Paddings or Dimension or None or Call:
                            cost = new CostModel.Cost() { [CostModel.CostFactorNames.CPUCycles] = 1 };
                            break;
                        case Op op:
                            {
                                if (!_rootSearchGraph.TryGetOutEdges(enode, out var edges))
                                {
                                    throw new NotSupportedException("graph doesn't contain the vertex.");
                                }

                                var tempArgs = edges.OrderBy(e => e.InputIndex).Select<CrossEdge, BaseExpr>(e => e.Target switch
                                {
                                    SearchableNode { Expr: Dimension attr } => attr,
                                    SearchableNode { Expr: Shape attr } => attr,
                                    SearchableNode { Expr: Padding attr } => attr,
                                    SearchableNode { Expr: Paddings attr } => attr,
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
                    if (enode.Expr is Op o && o is not Boxing)
                    {
                        clusterVarMemo[cluster].Add(boolVar);
                    }
                }
            }
        }

        // 1. must pick one in root enode.
        cpmodel.AddExactlyOne(rootCluster.Vertices.Select(n => varMemo[n]).ToArray());

        // 2. pick only one in each cluster.
        foreach (var (cluster, vars) in clusterVarMemo)
        {
            if (vars.Count > 0)
            {
                cpmodel.AddExactlyOne(vars.ToArray());
            }
        }

        // 3. when pick node, must pick one child node.
        foreach (var n in _rootSearchGraph.Vertices)
        {
            var ns = new[] { varMemo[n].Not() };

            if (_rootSearchGraph.TryGetOutEdges(n, out var allEdges))
            {
                foreach (var argEdges in allEdges.GroupBy(g => g.InputIndex))
                {
                    var cns = argEdges.SelectMany(e => e.InputGraph.Vertices).Select(cn => varMemo[cn]).ToList();
                    if (cns.Count > 0)
                    {
                        cpmodel.Add(LinearExpr.Sum(cns) == 1).OnlyEnforceIf(varMemo[n]);
                    }
                }
            }
        }

#if false
        // 4. no cycle
        foreach (var cluster in _rootSearchGraph.Clusters.OfType<DistributedSearchGraph>())
        {
            foreach (var sourceBucket in cluster.Clusters.OfType<DistributedSearchGraph>())
            {
                foreach (var destBucket in cluster.Clusters.OfType<DistributedSearchGraph>().Where(b => !ReferenceEquals(b, sourceBucket)))
                {
                    foreach (var (src, dest) in sourceBucket.Vertices.Where(v => v.IsBidirect).Zip(destBucket.Vertices.Where(v => v.IsBidirect)))
                    {
                        cpmodel.AddBoolAnd([varMemo[src].Not(), varMemo[dest].Not()]);
                    }
                }
            }
        }
#endif

        // 5. add pick weights for all enode.
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
#if true
        using (var stream = enableDump ? Diagnostics.DumpScope.Current.OpenFile("Costs/Pick.dot") : Stream.Null)
        {
            Dump(stream, picks, costMemo);
        }
#endif

        if (_phase == AutoDistributedPhase.SearchConstant)
        {
            foreach (var pick in picks)
            {
                if (pick.Value && pick.Key.Expr is TensorConst { ValueType: DistributedType } distConst
                    && _distributedConstSources.TryGetValue(distConst, out var source))
                {
                    DistributedConsts.Add(source, distConst);
                }
            }
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
    private readonly Dictionary<SearchableNode, BaseExpr> _memo;

    public ExprBuildVisitor(DistributedSearchGraph rootSearchGraph, Dictionary<SearchableNode, bool> picks)
    {
        _rootSearchGraph = rootSearchGraph;
        _picks = picks;
        _memo = new();
    }

    public BaseExpr Visit(IEnumerable<DistributedSearchGraph> rootBuckets)
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
                case Var or TensorConst or TupleConst or None or Shape or Padding or Paddings or Dimension or Call:
                    expr = root.Expr;
                    break;
                case BaseFunction func:
                    expr = new Call(target: func, arguments: children);
                    break;
                case Op op:
                    expr = new Call(target: op, arguments: children);
                    break;
                case IR.Tuple tp:
                    expr = (BaseExpr)tp.With(fields: children);
                    break;
                case IR.If @if:
                    expr = @if.With(condition: (Expr)children[^3], then: (BaseFunction)children[^2], @else: (BaseFunction)children[^1], arguments: children[..^3].ToArray());
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
    public DistributedCostEvaluateContext(Op op, IRType returnType, BaseExpr[] args, CompileOptions compileOptions)
    {
        Op = op;
        ReturnType = returnType;
        Args = args;
        CompileOptions = compileOptions;
    }

    public Op Op { get; }

    public IRType ReturnType { get; }

    public BaseExpr[] Args { get; }

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
