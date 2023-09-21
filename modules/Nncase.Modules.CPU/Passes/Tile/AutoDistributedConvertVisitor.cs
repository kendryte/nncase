// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using System.Runtime.CompilerServices;
using Nncase.IR;
using Nncase.IR.CPU;
using Nncase.IR.Tensors;

[assembly: InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.Passes.Tile;

internal sealed class AutoDistributedConvertVisitor : ExprVisitor<IReadOnlyList<Expr>, Unit>
{
    private readonly Dictionary<Expr, Expr> _originMap;

    public AutoDistributedConvertVisitor(TileOptions tileOptions)
    {
        TileOptions = tileOptions;
        Placement = new Placement(Placement.DeviceKind.CPU, tileOptions.Hierarchy, "bt");
        _originMap = new Dictionary<Expr, Expr>(ReferenceEqualityComparer.Instance);
    }

    public TileOptions TileOptions { get; }

    public Placement Placement { get; }

    public static IReadOnlyList<Expr> GetLeafCandidateBoxings(Expr expr, Placement placement)
    {
        return Utilities.DistributedUtility.GetLeafCandidateNDSBPs((TensorType)expr.CheckedType, placement).Select(ndsbp => IR.F.CPU.Boxing(expr, new DistributedType((TensorType)expr.CheckedType, ndsbp, placement))).ToArray();
    }

    /// <summary>
    /// when input expression sbp is partial, get the new candidate boxings.
    /// </summary>
    /// <param name="expr">input expression.</param>
    /// <returns>the boxings.</returns>
    /// <exception cref="NotSupportedException">when expr is tuple.</exception>
    public static IReadOnlyList<Expr> GetPartialCandidateBoxings(Expr expr)
    {
        if (expr is IR.Tuple tuple)
        {
            var candidates = tuple.Fields.ToArray().
                Select(GetPartialCandidateBoxings).
                CartesianProduct();
            return candidates.Any() ? candidates.
                Select(fs => new IR.Tuple(fs.ToArray())).
                ToArray() : Array.Empty<Expr>();
        }

        var type = (DistributedType)expr.CheckedType;
        if (!type.NdSBP.Any(sbp => sbp is SBPBroadCast))
        {
            return Array.Empty<Expr>();
        }

        var tensorType = type.TensorType;
        var candidateNdsbps = new List<SBP>[type.Placement.Rank];
        for (int i = 0; i < type.Placement.Rank; i++)
        {
            candidateNdsbps[i] = new List<SBP>();
            if (type.NdSBP[i] is SBPPartialSum)
            {
                candidateNdsbps[i].Add(SBP.B);
                for (int axis = 0; axis < tensorType.Shape.Rank; axis++)
                {
                    if (tensorType.Shape[axis] is { IsFixed: true, Value: int s } && Utilities.DistributedUtility.IsDivisible(s, type.Placement.Hierarchy[i]))
                    {
                        candidateNdsbps[i].Add(SBP.S(axis));
                    }
                }
            }
        }

        return candidateNdsbps.CartesianProduct().
            Select(ndsbp => new DistributedType(tensorType, new IRArray<SBP>(ndsbp), type.Placement)).
            Select(disttype => IR.F.CPU.Boxing(expr, disttype)).ToArray();
    }

    public Expr Convert(Expr body)
    {
        var equivalents = Visit(body).Select(newbody => IR.F.CPU.Boxing(newbody, body.CheckedType)).ToArray();
        var graph = new EGraph();
        var bodyEclasses = equivalents.Select(graph.Add).ToArray();
        foreach (var cls in bodyEclasses.Skip(1))
        {
            graph.Union(bodyEclasses[0], cls);
        }

        graph.Rebuild();
        return graph.Extract(bodyEclasses[0], null, out _);
    }

    protected override IReadOnlyList<Expr> DefaultVisitLeaf(Expr expr)
    {
        return Array.Empty<Expr>();
    }

    protected override IReadOnlyList<Expr> VisitLeafTuple(IR.Tuple expr)
    {
        return UpdateEquivalents(expr.Fields.ToArray().Select(Visit).CartesianProduct().Select(e => new IR.Tuple(e.ToArray())).ToArray(), expr);
    }

    protected override IReadOnlyList<Expr> VisitLeafCall(Call expr)
    {
        if (expr.Target is not Op op)
        {
            throw new NotSupportedException("not support auto distributed call function");
        }

        var equivalArgs = op.Parameters.
            Select(param => VisitLeafArgument(param.ParameterKind, expr.Arguments[param.Index])).ToArray();
        var candidateEquivalCalls = equivalArgs.
            CartesianProduct().
            Select(args => args.ToArray()).
            Select(args => BuildEquivalCalls(op, args)).
            SelectMany(i => i).
            ToArray();

        if (candidateEquivalCalls.Any(t => t.Valid))
        {
            return UpdateEquivalents(Canonicalize(candidateEquivalCalls.Where(t => t.Valid).Select(t => t.Call)), expr);
        }

        var boxingArgs = op.Parameters.Select(param => param.ParameterKind switch
            {
                ParameterKind.Input => ExprMemo[expr.Arguments[param.Index]].Select(GetReBoxings).SelectMany(i => i),
                ParameterKind.Attribute => ExprMemo[expr.Arguments[param.Index]],
                _ => throw new NotSupportedException(),
            }).ToArray();

        var candidateBoxingCalls = boxingArgs.
            CartesianProduct().
            Select(args => args.ToArray()).
            Select(args => new Call(op, args)).
            Select<Call, (bool Valid, Call Call)>(c => (c.InferenceType(), c)).
            ToArray();

        if (candidateBoxingCalls.Any(t => t.Valid))
        {
            return UpdateEquivalents(Canonicalize(candidateBoxingCalls.Where(t => t.Valid).Select(t => t.Call)), expr);
        }

        throw new InvalidDataException("after reboxing still can't infer!");
    }

    private IReadOnlyList<Expr> VisitLeafArgument(ParameterKind parameterKind, Expr expr)
    {
        IReadOnlyList<Expr> equivalents;
        if (!ExprMemo[expr].Any())
        {
            equivalents = (parameterKind, expr) switch
            {
                (ParameterKind.Input, Expr e) when e is Const or Var => GetLeafCandidateBoxings(e, Placement),
                (ParameterKind.Input, Expr e) when e is IR.Tuple tp => tp.Fields.ToArray().Select(f => VisitLeafArgument(parameterKind, f)).CartesianProduct().Select(e => new IR.Tuple(e.ToArray())).ToArray(),
                (ParameterKind.Attribute, Expr e) when e is Const or Var => new[] { e },
                _ => throw new InvalidOperationException(),
            };
            ExprMemo[expr] = equivalents;
            UpdateEquivalents(equivalents, expr);
        }
        else
        {
            equivalents = ExprMemo[expr];
        }

        return equivalents;
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
                if (tensorType.Shape[axis] is { IsFixed: true, Value: int s } && Utilities.DistributedUtility.IsDivisible(s, type.Placement.Hierarchy[i]))
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

    private IEnumerable<(bool Valid, Call Call)> BuildEquivalCalls(Op target, Expr[] args)
    {
        if (!target.Parameters.Where(p => p.ParameterKind == ParameterKind.Input).All(p => IsDistributed(args[p.Index].CheckedType)))
        {
            throw new InvalidDataException();
        }

        var calls = new List<(bool, Call)>();
        var call = new Call(target, args);
        var valid = call.InferenceType();
        calls.Add((valid, call));
        if (!valid)
        {
            if (target is CPUKernelOp { Target: Reshape } || target is Reshape)
            {
                // the reshape need force boxing.
                var newShape = ((TensorConst)args[1]).Value.ToArray<int>();
                var inType = (DistributedType)args[0].CheckedType;
                var tensorType = inType.TensorType with { Shape = newShape };
                calls.AddRange(Utilities.DistributedUtility.GetLeafCandidateNDSBPs(tensorType, inType.Placement).
                    Select(ndsbp => IR.F.CPU.Boxing(args[0], new DistributedType(tensorType, ndsbp, inType.Placement))).
                    Select(c => (c.InferenceType(), c)));
            }
            else
            {
                // when args have partial, we need boxing args.
                var broadcastArgs = args.Zip(target.Parameters).Select(t => t.Second.ParameterKind == ParameterKind.Input ? GetPartialCandidateBoxings(t.First) : Array.Empty<Expr>()).ToArray();

                if (!broadcastArgs.All(bargs => bargs.Count == 0))
                {
                    calls.AddRange(broadcastArgs.Select((bargs, i) => bargs.Any() ? UpdateEquivalents(bargs, _originMap[args[i]]) : bargs.Concat(new[] { args[i] })).
                        CartesianProduct().
                        Select(bargs => bargs.ToArray()).
                        Select(bargs => new Call(target, bargs)).
                        Select(c => (c.InferenceType(), c)));
                }
            }
        }

        return calls;
    }

    private IReadOnlyList<Expr> Canonicalize(IEnumerable<Call> candidateEquivalCalls)
    {
        var equivalCalls = new List<Expr>();
        foreach (var group in candidateEquivalCalls.GroupBy(c => c.CheckedType))
        {
            if (group.Count() > 1)
            {
                // extract group
                var egrph = new EGraph();
                var roots = group.Select(egrph.Add).ToArray();
                foreach (var r in roots.Skip(1))
                {
                    egrph.Union(roots[0], r);
                }

                egrph.Rebuild();
                var best = egrph.Extract(roots[0], null, out var picks);
                foreach (var (enode, picked) in picks)
                {
                    if (!picked && _originMap.TryGetValue(enode.Expr, out var originExpr))
                    {
                        ExprMemo[originExpr] = ExprMemo[originExpr].Where(e => !ReferenceEquals(e, enode.Expr)).ToArray();
                        _originMap.Remove(enode.Expr);
                    }
                }

                equivalCalls.Add(best);
            }
            else
            {
                equivalCalls.Add(group.First());
            }
        }

        return equivalCalls;
    }

    private bool IsDistributed(IRType type) => type switch
    {
        DistributedType => true,
        TupleType t => t.All(IsDistributed),
        _ => false,
    };

    private IReadOnlyList<Expr> UpdateEquivalents(IReadOnlyList<Expr> equivalents, Expr origin)
    {
        foreach (var item in equivalents)
        {
            _originMap[item] = origin;
        }

        return equivalents;
    }
}
