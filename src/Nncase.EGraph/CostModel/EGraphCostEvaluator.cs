// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Transform;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.CostModel;

internal sealed class EGraphCostEvaluator
{
    private readonly EClass _root;
    private readonly Dictionary<ENode, Cost> _costs = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<EClass, Cost> _eclassCosts = new();
    private readonly Dictionary<EClass, AccumulateSequence> _eclassAccumulated = new();
    private readonly HashSet<EClass> _allEclasses = new();
    private readonly IBaseFuncCostEvaluator? _baseFuncCostEvaluator;
    private bool _changed;

    public EGraphCostEvaluator(EClass root, IBaseFuncCostEvaluator? basefunc_cost_evaluator)
    {
        _root = root;
        _baseFuncCostEvaluator = basefunc_cost_evaluator;
        PopulateAllEclasses(_root);
    }

    public EGraphCostModel Evaluate()
    {
        while (true)
        {
            _changed = false;
            TryEvaluateAll();
            if (!_changed)
            {
                break;
            }
        }

        if (!_eclassCosts.ContainsKey(_root))
        {
            throw new InvalidOperationException("Cannot evaluate cost for root.");
        }

        return new(_costs);
    }

    private void PopulateAllEclasses(EClass eClass)
    {
        if (!_allEclasses.Contains(eClass))
        {
            _allEclasses.Add(eClass);
            foreach (var node in eClass.Nodes)
            {
                foreach (var child in node.Children)
                {
                    PopulateAllEclasses(child);
                }
            }
        }
    }

    private void TryEvaluateAll()
    {
        foreach (var eclass in _allEclasses)
        {
            Visit(eclass);
        }
    }

    private Cost? Visit(EClass eclass)
    {
        Cost? cost = null;
        ENode? minCostEnode = null;
        foreach (var enode in eclass.Nodes)
        {
            var newCost = Visit(enode, eclass.CheckedType);
            if (newCost != null
                && (cost == null || newCost < cost))
            {
                cost = newCost;
                minCostEnode = enode;
            }
        }

        return UpdateCost(eclass, minCostEnode, cost);
    }

    private Cost? Visit(ENode enode, IRType returnType)
    {
        return enode.Expr switch
        {
            Var var => Visit(enode, var),
            TensorConst con => Visit(enode, con),
            TupleConst con => Visit(enode, con),
            Function func => Visit(enode, func),
            Call call => Visit(enode, call, returnType),
            IR.Tuple tuple => Visit(enode, tuple),
            Op op => Visit(enode, op),
            Marker marker => Visit(enode, marker),
            None none => Visit(enode, none),
            BaseFunction baseFunction => Visit(enode, baseFunction),
            _ => throw new ArgumentException("Unsupported expression type."),
        };
    }

    private Cost Visit(ENode enode, Var var)
    {
        return VisitLeaf(enode, () => Cost.Zero);
    }

    private Cost Visit(ENode enode, TensorConst tc)
    {
        return VisitLeaf(enode, () => Cost.Zero);
    }

    private Cost Visit(ENode enode, Op op)
    {
        return VisitLeaf(enode, () => Cost.Zero);
    }

    private Cost? Visit(ENode enode, Function func)
    {
        return Visit(enode, costs => Cost.Zero);
    }

    private Cost? Visit(ENode enode, TupleConst tc)
    {
        return Visit(enode, costs => Cost.Zero);
    }

    private Cost? Visit(ENode enode, IR.Tuple tuple)
    {
        return Visit(enode, costs => AccumulateCosts(costs));
    }

    private Cost? Visit(ENode enode, Marker marker)
    {
        return Visit(enode, costs => costs[0].Cost);
    }

    private Cost? Visit(ENode enode, None none)
    {
        return Visit(enode, costs => Cost.Zero);
    }

    private Cost? Visit(ENode enode, Call call, IRType returnType)
    {
        return Visit(enode, costs =>
        {
            Cost? targetCost = null;
            foreach (var targetEnode in enode.Children[0].Nodes)
            {
                Cost? newTargetCost;
                if (targetEnode.Expr is Op op)
                {
                    var context = new EGraphOpCostEvaluateContext(returnType, enode.Children.Skip(1).Select(x => x.CheckedType).ToArray());
                    newTargetCost = CompilerServices.EvaluateOpCost(op, context);
                }
                else
                {
                    // Trace.Assert(targetEnode.Expr is Function);
                    newTargetCost = Visit(targetEnode, returnType);
                }

                if (targetCost == null || (newTargetCost != null && newTargetCost < targetCost))
                {
                    targetCost = newTargetCost;
                }
            }

            return UpdateCost(enode, targetCost == null ? null : targetCost + AccumulateCosts(costs));
        });
    }

    private Cost? Visit(ENode enode, BaseFunction baseFunction)
    {
        return VisitLeaf(enode, () => _baseFuncCostEvaluator!.VisitLeaf(baseFunction));
    }

    private Cost? UpdateCost(ENode enode, Cost? cost)
    {
        if (_costs.TryGetValue(enode, out var oldCost))
        {
            if (oldCost != cost)
            {
                if (cost == null)
                {
                    _costs.Remove(enode);
                }
                else
                {
                    _costs[enode] = cost;
                }

                _changed = true;
            }
        }
        else
        {
            if (cost != null)
            {
                _costs.Add(enode, cost);
                _changed = true;
            }
        }

        return cost;
    }

    private Cost? UpdateCost(EClass eclass, ENode? minCostEnode, Cost? cost)
    {
        if (_eclassCosts.TryGetValue(eclass, out var oldCost))
        {
            if (oldCost != cost)
            {
                if (cost == null)
                {
                    _eclassCosts.Remove(eclass);
                    _eclassAccumulated.Remove(eclass);
                }
                else
                {
                    _eclassCosts[eclass] = cost;
                    var acc = _eclassAccumulated[eclass];
                    acc.Current = eclass;
                    acc.Accumulated.Clear();
                    foreach (var child in minCostEnode!.Children)
                    {
                        acc.Accumulated.Add(_eclassAccumulated[child].Current);
                        acc.Accumulated.UnionWith(_eclassAccumulated[child].Accumulated);
                    }
                }

                _changed = true;
            }
        }
        else
        {
            if (cost != null)
            {
                _eclassCosts.Add(eclass, cost);
                _changed = true;
                var acc = new AccumulateSequence(eclass);
                foreach (var child in minCostEnode!.Children)
                {
                    acc.Accumulated.Add(_eclassAccumulated[child].Current);
                    acc.Accumulated.UnionWith(_eclassAccumulated[child].Accumulated);
                }

                _eclassAccumulated[eclass] = acc;
            }
        }

        return cost;
    }

    private Cost VisitLeaf(ENode enode, Func<Cost> costGetter)
    {
        if (!_costs.TryGetValue(enode, out var cost))
        {
            cost = costGetter();
            _costs[enode] = cost;
            _changed = true;
        }

        return cost;
    }

    private Cost? Visit(ENode enode, Func<EClassCost[], Cost?> costGetter)
    {
        var costs = new EClassCost[enode.Children.Count];
        for (int i = 0; i < costs.Length; i++)
        {
            var child = enode.Children[i];
            if (_eclassCosts.TryGetValue(child, out var childCost))
            {
                costs[i] = new(child, childCost, _eclassAccumulated[enode.Children[i]]);
            }
            else
            {
                return null;
            }
        }

        var cost = costGetter(costs);
        return UpdateCost(enode, cost);
    }

    private Cost AccumulateCosts(EClassCost[] costs)
    {
        HashSet<EClass> candidates = new(costs.Select(c => c.Sequence.Current));

        foreach (var item in costs)
        {
            candidates.ExceptWith(item.Sequence.Accumulated);
        }

        return candidates.Aggregate(Cost.Zero, (acc, e) => acc + _eclassCosts[e]);
    }

    private sealed record AccumulateSequence
    {
        public EClass Current;

        public AccumulateSequence(EClass current)
        {
            Current = current;
            Accumulated = new();
        }

        public readonly HashSet<EClass> Accumulated;
    }

    private sealed record EClassCost(EClass Class, Cost Cost, AccumulateSequence Sequence)
    {
    }

    private sealed class EGraphOpCostEvaluateContext : ICostEvaluateContext
    {
        private readonly IRType? _returnType;
        private readonly IRType?[] _argumentTypes;

        public EGraphOpCostEvaluateContext(IRType? returnType, IRType?[] argumentTypes)
        {
            _returnType = returnType;
            _argumentTypes = argumentTypes;
        }

        public T GetArgumentType<T>(Op op, ParameterInfo parameter)
            where T : IRType
        {
            if (op.GetType() == parameter.OwnerType)
            {
                return (T?)_argumentTypes[parameter.Index] ?? throw new InvalidOperationException("Run type infer first.");
            }
            else
            {
                throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
            }
        }

        public T GetReturnType<T>()
            where T : IRType
        {
            return (T?)_returnType ?? throw new InvalidOperationException("Run type infer first.");
        }
    }
}
