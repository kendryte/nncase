// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Nncase.CostModel;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase.CostModel;

internal sealed class EGraphCostEvaluator
{
    private readonly EClass _root;
    private readonly Dictionary<ENode, Cost> _costs = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<EClass, Cost> _eclassCosts = new();
    private readonly HashSet<EClass> _allEclasses = new();
    private readonly IBaseFuncCostEvaluator? _baseFuncCostEvaluator;
    private readonly bool _accumulate;
    private bool _changed;

    public EGraphCostEvaluator(EClass root, IBaseFuncCostEvaluator? basefunc_cost_evaluator, bool accumulate = true)
    {
        _root = root;
        _accumulate = accumulate;
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
        foreach (var enode in eclass.Nodes)
        {
            var newCost = Visit(enode, eclass.CheckedType);
            if (newCost != null
                && (cost == null || newCost < cost))
            {
                cost = newCost;
            }
        }

        return UpdateCost(eclass, cost);
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
            If @if => Visit(enode, @if),
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
        return Visit(enode, costs => costs.Sum());
    }

    private Cost? Visit(ENode enode, IR.Tuple tuple)
    {
        return Visit(enode, costs => _accumulate ? costs.Sum() : Cost.Zero);
    }

    private Cost? Visit(ENode enode, If @if)
    {
        return Visit(enode, cost => _accumulate ? cost[^3] + cost[^2] + cost[^1] : Cost.Zero);
    }

    private Cost? Visit(ENode enode, Marker marker)
    {
        return Visit(enode, costs => _accumulate ? costs[0] : Cost.Zero);
    }

    private Cost? Visit(ENode enode, None none)
    {
        return Visit(enode, costs => Cost.Zero);
    }

    private Cost? Visit(ENode enode, Call call, IRType returnType)
    {
        return Visit(enode, costs =>
        {
            Cost? cost = null;
            foreach (var targetEnode in enode.Children[0].Nodes)
            {
                Cost? newCost;
                if (targetEnode.Expr is Op op)
                {
                    var context = new EGraphOpCostEvaluateContext(returnType, enode.Children.Skip(1).Select(x => x.CheckedType).ToArray(), enode.Children.Skip(1).ToArray());
                    newCost = CompilerServices.EvaluateOpCost(op, context);
                }
                else
                {
                    // Trace.Assert(targetEnode.Expr is Function);
                    newCost = Visit(targetEnode, returnType);
                }

                if (cost == null || (newCost != null && newCost < cost))
                {
                    cost = newCost;
                }
            }

            return UpdateCost(enode, cost == null ? null : (_accumulate ? cost + costs.Sum() : cost));
        });
    }

    private Cost? Visit(ENode enode, BaseFunction baseFunction)
    {
        return VisitLeaf(enode, () => _baseFuncCostEvaluator?.VisitLeaf(baseFunction) ?? Cost.Zero);
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

    private Cost? UpdateCost(EClass eclass, Cost? cost)
    {
        if (_eclassCosts.TryGetValue(eclass, out var oldCost))
        {
            if (oldCost != cost)
            {
                if (cost == null)
                {
                    _eclassCosts.Remove(eclass);
                }
                else
                {
                    _eclassCosts[eclass] = cost;
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

    private Cost? Visit(ENode enode, Func<Cost[], Cost?> costGetter)
    {
        var costs = new Cost[enode.Children.Count];
        for (int i = 0; i < costs.Length; i++)
        {
            if (_eclassCosts.TryGetValue(enode.Children[i], out var childCost))
            {
                costs[i] = childCost;
            }
            else
            {
                return null;
            }
        }

        var cost = costGetter(costs);
        return UpdateCost(enode, cost);
    }
}

internal sealed class EGraphOpCostEvaluateContext : ICostEvaluateContext
{
    private readonly IRType? _returnType;
    private readonly IRType?[] _argumentTypes;
    private readonly EClass[] _arguments;

    public EGraphOpCostEvaluateContext(IRType? returnType, IRType?[] argumentTypes, EClass[] arguments)
    {
        _returnType = returnType;
        _argumentTypes = argumentTypes;
        _arguments = arguments;
    }

    public T GetArgument<T>(Op op, ParameterInfo parameter)
      where T : BaseFunction
    {
        System.Diagnostics.Trace.Assert(_arguments[parameter.Index].Nodes.Count == 1);
        return (T)_arguments[parameter.Index].Nodes[0].Expr;
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
