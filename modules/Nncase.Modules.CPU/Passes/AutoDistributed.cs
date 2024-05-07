// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase.CodeGen;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.PatternMatch.Utility;

[assembly: InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.Passes;

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
        var rewriter = new AutoDistributedRewriter(_compileOptions, _compileOptions.TargetCompileOptions is CpuTargetOptions options ? options : new CpuTargetOptions());
        return Task.FromResult(rewriter.Rewirte(input));
    }
}

internal sealed class AutoDistributedRewriter : ExprVisitor<Dictionary<IRType, List<Expr>>, Unit>
{
    public AutoDistributedRewriter(CompileOptions compileOptions, CpuTargetOptions targetOptions)
    {
        Placement = new Placement(targetOptions.Hierarchy, targetOptions.HierarchyNames);
        CompileOptions = compileOptions;
        TargetOptions = targetOptions;
    }

    public Placement Placement { get; }

    public CompileOptions CompileOptions { get; }

    public CpuTargetOptions TargetOptions { get; }

    public static IReadOnlyList<Expr> GetLeafCandidateBoxings(Expr expr, Placement placement)
    {
        return Utilities.DistributedUtility.GetLeafCandidateNDSBPs((TensorType)expr.CheckedType, placement).
            Select(ndsbp => IR.F.CPU.Boxing(expr, new DistributedType((TensorType)expr.CheckedType, ndsbp, placement))).
            ToArray();
    }

    public BaseFunction Rewirte(BaseFunction input)
    {
        if (input is Function function)
        {
            var equivalents = Visit(function.Body).Select(g => InstertTerminator(g.Value[0])).ToArray();
            using (new ExprPinner(equivalents))
            {
                BranchCut();
            }

            var graph = new EGraph();
            foreach (var (exprKey, buckets) in ExprMemo.Where(kv => kv.Key is not Op))
            {
                foreach (var (typeKey, bucket) in buckets.Where(kv => kv.Value.Any()))
                {
                    Unions(graph, bucket);
                }
            }

            var root = Unions(graph, equivalents);
            var post = graph.Extract(root, CompileOptions, null, Array.Empty<EGraphExtractConstrains>());
            return function.With(body: post);
        }

        return input;
    }

    protected override Dictionary<IRType, List<Expr>> DefaultVisitLeaf(Expr expr)
    {
        return new();
    }

    protected override Dictionary<IRType, List<Expr>> VisitLeafTuple(IR.Tuple expr)
    {
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
            throw new NotSupportedException("not support auto distributed call function");
        }

        var isSupported = PassUtility.IsCpuSupported(op, expr.Arguments.ToArray());
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
                    ToDictionary(g => g.Key, g => new List<Expr>(g.ToList<Expr>()));

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
                    ToDictionary(g => g.Key, g => new List<Expr>(g.ToList<Expr>()));
        }

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
        };

        var buckets = ExprMemo[expr];
        if (!buckets.Any())
        {
            switch (parameterKind, expr)
            {
                case (ParameterKind.Input, Expr e) when e is Const or Var:
                    updateBuckets(buckets, isSupported ? GetLeafCandidateBoxings(e, Placement) : new[] { e });
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
                    var results = buckets.Select(kv => GetLeafCandidateBoxings(kv.Value[0], Placement)).SelectMany(i => i).ToArray();
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
            if (call.CheckedType is DistributedType distributedType)
            {
                calls.AddRange(Utilities.DistributedUtility.GetPartialCandidateNDSBPs(distributedType).
                    Select(ndsbp => IR.F.CPU.Boxing(call, distributedType with { NdSBP = ndsbp })));
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
                var boxingP2B = IR.F.CPU.Boxing(e, new DistributedType(type.TensorType, type.NdSBP.Select(s => s is SBPPartialSum ? SBP.B : s).ToArray(), Placement));
                return IR.F.CPU.Boxing(boxingP2B, type.TensorType);
            }

            return IR.F.CPU.Boxing(e, type.TensorType);
        }

        return (expr, expr.CheckedType) switch
        {
            (IR.Tuple tp, TupleType tptype) => new IR.Tuple(tp.Fields.ToArray().Select(InstertTerminator).ToArray()),
            (Expr e, DistributedType type) => CreateFinalBoxing(e, type),
            (Expr e, TensorType type) => e,
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
            foreach (var (_, bukets) in ExprMemo)
            {
                foreach (var (_, buket) in bukets.Where(kv => kv.Value.Any()))
                {
                    if (!buket[0].Users.Any())
                    {
                        foreach (var item in buket)
                        {
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
