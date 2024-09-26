// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using System.Runtime.CompilerServices;
using Google.OrTools.Sat;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Targets;
using Nncase.Utilities;
using static Nncase.PatternMatch.Utility;

[assembly: InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.Passes.Distributed;

public interface IEquality
{
}

public record EqualityNode(Expr Expr) : IEquality
{
}

public record EqualityClass(bool Tuple, List<IEquality> Children) : IEquality
{
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

internal sealed class AutoDistributedRewriter : ExprVisitor<Dictionary<IRType, List<Expr>>, Unit>
{
    private readonly Dictionary<Expr, IEquality> _equalMemo = new();

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

    public static IReadOnlyList<Expr> GetLeafCandidateBoxings(Expr expr, IEnumerable<Placement> placements)
    {
        return placements.Select(
            placement =>
                Utilities.DistributedUtility.GetLeafCandidateNDSBPs((TensorType)expr.CheckedType, placement).
                Select(ndsbp =>
                    IR.F.CPU.Boxing(expr, new DistributedType((TensorType)expr.CheckedType, ndsbp, placement)))).
            SelectMany(e => e).ToArray();
    }

    public static IReadOnlyList<IRArray<SBP>> GetDiverseCandidateSBPs(DistributedType distributedType, IEnumerable<Placement> placements)
    {
        return placements.Select(
            placement =>
                Utilities.DistributedUtility.GetLeafCandidateNDSBPs(distributedType.TensorType, placement).
                Where(ndsbp => ndsbp != distributedType.NdSBP)).
            SelectMany(e => e).ToArray();
    }

    public void SingleNodeMemoryExtractConstrains(CpModel model, IReadOnlyDictionary<ENode, BoolVar> vars)
    {
        var distTypes = vars.Keys.Where(k => k.Expr.CheckedType is DistributedType dt).ToArray();
        foreach (var k in distTypes)
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

    public void FilterByScheme(Expr expr, Dictionary<IRType, List<Expr>> result)
    {
        foreach (var name in expr.Metadata.OutputNames ?? Array.Empty<string>())
        {
            if (Scheme.TryGetValue(name, out var tp))
            {
                var keys = result.Keys.ToArray();
                foreach (var key in keys)
                {
                    if (!(key is DistributedType dtype && dtype.NdSBP == tp.NdSBP && dtype.Placement == tp.Placement))
                    {
                        result.Remove(key);
                    }
                }
            }
        }
    }

    public BaseFunction Rewirte(BaseFunction input)
    {
        if (input is Function function)
        {
            var typeEquivalents = Visit(function.Body);

            if (function.Body is IR.Tuple tp)
            {
                var outputs = new List<Expr>();
                var equ = _equalMemo[tp];

                void Dfs(IEquality equality)
                {
                    switch (equality)
                    {
                        case EqualityNode n:
                            outputs.Add(n.Expr);
                            break;
                        case EqualityClass tp:
                            foreach (var item in tp.Children)
                            {
                                Dfs(item);
                            }

                            break;
                    }
                }

                Dfs(equ);

                using (new ExprPinner(outputs.ToArray()))
                {
                    BranchCut();
                }
            }
            else
            {
                var outputs = typeEquivalents.Select(g => InstertTerminator(g.Value[0]))
                .Select(e => new EqualityNode(e))
                .OfType<IEquality>().ToList();
                _equalMemo.Add(function.Body, new EqualityClass(false, outputs));

                using (new ExprPinner(outputs.Select(e => ((EqualityNode)e).Expr).ToArray()))
                {
                    BranchCut();
                }
            }

            var graph = new EGraph();
            foreach (var (exprKey, buckets) in ExprMemo.Where(kv => kv.Key is not Op))
            {
                foreach (var (typeKey, bucket) in buckets.Where(kv => kv.Value.Any()))
                {
                    Unions(graph, bucket);
                }
            }

            var equivalents = _equalMemo[function.Body];
            EClass Ddfs(IEquality equival)
            {
                switch (equival)
                {
                    case EqualityNode n:
                        return graph.Add(n.Expr);
                    case EqualityClass tp:
                        var eids = tp.Children.Select(Ddfs).ToArray();
                        if (tp.Tuple)
                        {
                            return graph.AddENode(new IR.Tuple(), eids);
                        }
                        else
                        {
                            foreach (var cls in eids.Skip(1))
                            {
                                graph.Union(eids[0], cls);
                            }

                            graph.Rebuild();
                            return eids[0];
                        }

                    default:
                        throw new NotSupportedException();
                }
            }

            var root = Ddfs(equivalents);
#if DEBUG
            using (var stream = Diagnostics.DumpScope.Current.OpenFile("egraph.dot"))
            {
                EGraphPrinter.DumpEgraphAsDot(graph, stream);
            }
#endif

            var constrains = new EGraphExtractConstrains[] { SingleNodeMemoryExtractConstrains };
            var post = graph.Extract(root, CompileOptions, null, constrains);
            return function.With(body: post);
        }

        return input;
    }

    protected override Dictionary<IRType, List<Expr>> DefaultVisitLeaf(Expr expr)
    {
        return new();
    }

    protected override Dictionary<IRType, List<Expr>> VisitLeafIf(If expr)
    {
        return new() { { expr.CheckedType, new() { expr } } };
    }

    protected override Dictionary<IRType, List<Expr>> VisitLeafTuple(IR.Tuple expr)
    {
        if (ReferenceEquals(expr, VisitRoot))
        {
            var fileds = new List<IEquality>();
            foreach (var i in Enumerable.Range(0, expr.Fields.Length))
            {
                var boxings = Visit(expr.Fields[i]).Values.
                    Select(l => l.Select(e => IR.F.CPU.Boxing(e, ((DistributedType)e.CheckedType).TensorType))).
                    SelectMany(e => e).Select(e => new EqualityNode(e)).OfType<IEquality>().ToList();
                fileds.Add(new EqualityClass(false, boxings));
            }

            _equalMemo.Add(expr, new EqualityClass(true, fileds));
            return new Dictionary<IRType, List<Expr>> { }; // return empty.
        }

        return expr.Fields.ToArray().
                Select(Visit).
                CartesianProduct().
                Select(e => new IR.Tuple(e.Select(e => e.Value[0]).ToArray())).
                GroupBy(tp => tp.CheckedType).
                ToDictionary(g => g.Key, g => g.ToList<Expr>());
    }

    protected override Dictionary<IRType, List<Expr>> VisitLeafCall(Call expr)
    {
        if (expr.Target is not Op op)
        {
            return new Dictionary<IRType, List<Expr>> { { expr.CheckedType, new() { expr } } };
        }

        var isSupported = PassUtility.IsCpuSupported(op, expr, expr.Arguments.ToArray());
        foreach (var param in op.Parameters)
        {
            VisitLeafArgument(param.ParameterKind, expr.Arguments[param.Index], isSupported);
        }

        var results = expr.Arguments.ToArray().
                    Select(Visit).
                    CartesianProduct().
                    Select(args => args.ToArray()).
                    Select(args => isSupported ? BuildEquivalCalls(op, args.Select(kv => kv.Value[0]).ToArray()).ToArray() :
                                    BuildNotSupportedCalls(op, args.Select(kv => kv.Value[0]).ToArray())).
                    SelectMany(i => i).
                    GroupBy(c => c.CheckedType).
                    ToDictionary(g => g.Key, g => g.OrderByDescending(e => e.Users.Count()).ToList<Expr>());

        if (results.Count == 0)
        {
            return expr.Arguments.ToArray().
                    Select(Visit).
                    CartesianProduct().
                    Select(args => args.ToArray()).
                    Select(args => new[] { new Call(op, args.Select(kv => kv.Value[0]).Select(arg => arg.CheckedType switch
                    {
                        DistributedType d => d.NdSBP.All(sbp => sbp is SBPBroadCast) ? arg : IR.F.CPU.Boxing(arg, d with { NdSBP = new(Enumerable.Repeat(SBP.B, d.NdSBP.Count)) }),
                        _ => arg,
                    }).ToArray()), }).
                    SelectMany(i => i).
                    GroupBy(c => c.CheckedType).
                    ToDictionary(g => g.Key, g => g.OrderByDescending(e => e.Users.Count()).ToList<Expr>());
        }

        if (expr.Target is not ScatterND && expr.Target is not Boxing && !expr.CheckedShape.ToValueArray().Contains(0) && results.Count == 1 && results.First().Key is DistributedType dt && dt.NdSBP.All(sbp => sbp is SBPBroadCast))
        {
            return expr.Arguments.ToArray().
                    Select(Visit).
                    CartesianProduct().
                    Select(args => args.ToArray()).
                    Select(args => args.Select(kv => kv.Value[0]).Select(arg => arg.CheckedType switch
                    {
                        DistributedType d => GetDiverseCandidateSBPs(d, Placements).Select(ndsbp => IR.F.CPU.Boxing(arg, new DistributedType(d.TensorType, ndsbp, d.Placement))).Concat(new[] { arg }).ToArray(),
                        _ => new[] { arg },
                    }).ToList().CartesianProduct().Select(arg => BuildEquivalCalls(op, arg.ToArray())).SelectMany(i => i).ToArray()).
                    SelectMany(i => i).
                    GroupBy(c => c.CheckedType).
                    ToDictionary(g => g.Key, g => g.OrderByDescending(e => e.Users.Count()).ToList<Expr>());
        }

        FilterByScheme(expr, results);
        return results;
    }

    private Dictionary<IRType, List<Expr>> VisitLeafArgument(ParameterKind parameterKind, Expr expr, bool isSupported)
    {
        var updateBuckets = (Dictionary<IRType, List<Expr>> buckets, IEnumerable<Expr> equivalents) =>
        {
            foreach (var eq in equivalents)
            {
                if (!buckets.TryGetValue(eq.CheckedType, out var bucket))
                {
                    bucket = new();
                    buckets.Add(eq.CheckedType, bucket);
                }

                bucket.Add(eq);
            }

            FilterByScheme(expr, buckets);
        };

        var buckets = ExprMemo[expr];
        if (!buckets.Any())
        {
            switch (parameterKind, expr)
            {
                case (ParameterKind.Input, Expr e) when e is Const or Var:
                    updateBuckets(buckets, isSupported ? GetLeafCandidateBoxings(e, Placements) : new[] { e });
                    break;
                case (ParameterKind.Input, Expr e) when e is IR.Tuple tp:
                    foreach (var f in tp.Fields)
                    {
                        VisitLeafArgument(parameterKind, f, isSupported);
                    }

                    foreach (var (k, v) in VisitLeafTuple(tp))
                    {
                        buckets.Add(k, v);
                    }

                    break;
                case (ParameterKind.Attribute, Var e):
                    updateBuckets(buckets, new[] { e });
                    break;
                case (ParameterKind.Attribute, TensorConst e):
                    updateBuckets(buckets, new[] { e.With() }); // remove all old users.
                    break;
                case (ParameterKind.Attribute, None e):
                    updateBuckets(buckets, new[] { e.With() });
                    break;
                default:
                    throw new InvalidOperationException();
            }
        }
        else if (parameterKind == ParameterKind.Input)
        {
            if (isSupported)
            {
                if (!buckets.Keys.Any(IsDistributed))
                {
                    var results = buckets.Select(kv => GetLeafCandidateBoxings(kv.Value[0], Placements)).SelectMany(i => i).ToArray();
                    updateBuckets(buckets, results);
                }
            }
            else
            {
                if (buckets.Keys.All(IsDistributed))
                {
                    var results = buckets.Select(kv => InstertTerminator(kv.Value[0])).ToArray();
                    updateBuckets(buckets, results);
                }
            }
        }

        return buckets;
    }

    private Call[] BuildNotSupportedCalls(Op target, Expr[] args)
    {
        if (target.Parameters.Where(p => p.ParameterKind == ParameterKind.Input).Any(p => IsDistributed(args[p.Index].CheckedType)))
        {
            return Array.Empty<Call>();
        }

        return new[] { new Call(target, args) };
    }

    private IEnumerable<Call> BuildEquivalCalls(Op target, Expr[] args)
    {
        if (!target.Parameters.Where(p => p.ParameterKind == ParameterKind.Input).All(p => IsDistributed(args[p.Index].CheckedType)))
        {
            return Array.Empty<Call>();
        }

        var calls = new List<Call>();
        var call = new Call(target, args);
        var valid = call.InferenceType();
        if (!valid)
        {
            // 1. dispose current call
            using var pinner = new ExprPinner(args);
            call.Dispose();

            if (target is Reshape)
            {
                // the reshape need force boxing.
                var newShape = ((TensorConst)args[1]).Value.ToArray<int>();
                var inType = (DistributedType)args[0].CheckedType;
                var tensorType = inType.TensorType with { Shape = newShape };
                foreach (var boxing in Utilities.DistributedUtility.GetLeafCandidateNDSBPs(tensorType, inType.Placement).
                    Select(ndsbp => IR.F.CPU.Boxing(args[0], new DistributedType(tensorType, ndsbp, inType.Placement))))
                {
                    if (boxing.CheckedType is InvalidType)
                    {
                        boxing.Dispose();
                    }
                    else
                    {
                        calls.Add(boxing);
                    }
                }
            }
            else
            {
                // todo expand search space.
                // calls.AddRange(Utilities.DistributedUtility.GetLeafCandidateNDSBPs(tensorType, inType.Placement).
                // Select(ndsbp => IR.F.CPU.Boxing(args[0], new DistributedType(tensorType, ndsbp, inType.Placement))));
            }
        }
        else
        {
            calls.Add(call);
            if (call.CheckedType is DistributedType distType)
            {
                // boxing for partialsum
                var partialBoxings = Utilities.DistributedUtility.GetPartialCandidateNDSBPs(distType).
                    Select(ndsbp => IR.F.CPU.Boxing(call, distType with { NdSBP = ndsbp })).ToArray();
                calls.AddRange(partialBoxings);

                using var pinner = new ExprPinner(calls.ToArray());
                var getExtraBoxings = (Expr expr) => Placements.
                    Where(p => p != distType.Placement).
                    Select(p => Utilities.DistributedUtility.GetLeafCandidateNDSBPs(distType.TensorType, p).
                        Select(ndsbp => IR.F.CPU.Boxing(expr, new DistributedType(distType.TensorType, ndsbp, p)))).
                    SelectMany(b => b);

                // boxing for other placements
                var extraBoxings = partialBoxings.Any() ? partialBoxings.Select(getExtraBoxings).SelectMany(i => i) : getExtraBoxings(call);
                foreach (var boxing in extraBoxings)
                {
                    if (boxing.CheckedType is InvalidType)
                    {
                        boxing.Dispose();
                    }
                    else
                    {
                        calls.Add(boxing);
                    }
                }
            }
        }

        return calls;
    }

    private IReadOnlyList<Expr> GetReBoxings(Expr expr)
    {
        if (expr is IR.Tuple tuple)
        {
            var candidates = tuple.Fields.ToArray().
                Select(GetReBoxings).
                CartesianProduct();
            return candidates.Any() ? candidates.
                Select(fs => new IR.Tuple(fs.ToArray())).
                ToArray() : Array.Empty<Expr>();
        }

        var type = (DistributedType)expr.CheckedType;
        var tensorType = type.TensorType;
        var candidateNdsbps = new List<SBP>[type.Placement.Rank];
        for (int i = 0; i < type.Placement.Rank; i++)
        {
            candidateNdsbps[i] = new List<SBP> { SBP.B };
            for (int axis = 0; axis < tensorType.Shape.Rank; axis++)
            {
                if (tensorType.Shape[axis] is { IsFixed: true, Value: int s } && Utilities.DistributedUtility.IsDivideExactly(s, type.Placement.Hierarchy[i]))
                {
                    candidateNdsbps[i].Add(SBP.S(axis));
                }
            }
        }

        return candidateNdsbps.CartesianProduct().
            Select(ndsbp => new IRArray<SBP>(ndsbp)).
            Where(ndsbp => ndsbp != type.NdSBP).
            Select(ndsbp => new DistributedType(tensorType, new IRArray<SBP>(ndsbp), type.Placement)).
            Select(disttype => IR.F.CPU.Boxing(expr, disttype)).ToArray();
    }

    private bool IsDistributed(IRType type) => type switch
    {
        DistributedType => true,
        TupleType t => t.All(IsDistributed),
        _ => false,
    };

    private Expr InstertTerminator(Expr expr)
    {
        Expr CreateFinalBoxing(Expr e, DistributedType type)
        {
            if (type.NdSBP.Any(s => s is SBPPartialSum))
            {
                var boxingP2B = IR.F.CPU.Boxing(e, new DistributedType(type.TensorType, type.NdSBP.Select(s => s is SBPPartialSum ? SBP.B : s).ToArray(), type.Placement));
                return IR.F.CPU.Boxing(boxingP2B, type.TensorType);
            }

            return IR.F.CPU.Boxing(e, type.TensorType);
        }

        return (expr, expr.CheckedType) switch
        {
            (IR.Tuple tp, TupleType tptype) => new IR.Tuple(tp.Fields.ToArray().Select(InstertTerminator).ToArray()),
            (Expr e, DistributedType type) => CreateFinalBoxing(e, type),
            (Expr e, TensorType type) => e,
            (Expr e, AnyType type) => e,
            (_, _) => throw new NotSupportedException(),
        };
    }

    private EClass Unions(EGraph graph, IEnumerable<Expr> equivalents)
    {
        var eids = equivalents.Select(graph.Add).ToArray();
        foreach (var cls in eids.Skip(1))
        {
            graph.Union(eids[0], cls);
        }

        graph.Rebuild();
        return eids[0];
    }

    private void BranchCut()
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            foreach (var (e, bukets) in ExprMemo)
            {
                foreach (var (_, buket) in bukets.Where(kv => kv.Value.Any()))
                {
                    if (!buket[0].Users.Any())
                    {
                        foreach (var item in buket)
                        {
                            if (item.Users.Any())
                            {
                                throw new InvalidOperationException("this item can't have more than zero users!");
                            }

                            using (new ExprPinner(item.Operands.ToArray()))
                            {
                                item.Dispose();
                            }
                        }

                        buket.Clear();
                        changed = true;
                    }
                }
            }
        }
    }
}
