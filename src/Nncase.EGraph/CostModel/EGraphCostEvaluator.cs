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

namespace Nncase.CostModel;

internal sealed class EGraphCostEvaluator
{
    private readonly IReadOnlyDictionary<EClass, List<ENode>> _eclasses;
    private readonly EClass _root;
    private readonly Dictionary<ENode, Cost> _costs = new();
    private bool _changed;

    public EGraphCostEvaluator(IReadOnlyDictionary<EClass, List<ENode>> eclasses, EClass root)
    {
        _eclasses = eclasses;
        _root = root;
    }

    public EGraphCostModel Evaluate()
    {
        while (true)
        {
            if (Visit(_root) != null && !_changed)
            {
                break;
            }
        }

        return new(_costs);
    }

    private Cost? Visit(EClass eclass)
    {
        Cost? cost = null;
        foreach (var enode in _eclasses[eclass])
        {
            var newCost = Visit(enode);
            if (cost == null || (newCost != null && newCost < cost))
            {
                cost = newCost;
            }
        }

        return cost;
    }

    private Cost? Visit(ENode enode)
    {
        return enode.Expr switch
        {
            Var var => Visit(enode, var),
            TensorConst con => Visit(enode, con),
            TupleConst con => Visit(enode, con),
            Function func => Visit(enode, func),
            Call call => Visit(enode, call),
            IR.Tuple tuple => Visit(enode, tuple),
            Op op => Visit(enode, op),
            _ => throw new ArgumentException("Unsupported expression type."),
        };
    }

    private Cost Visit(ENode enode, Var var)
    {
        return VisitLeaf(enode, () => Cost.Zero);
    }

    private Cost Visit(ENode enode, TensorConst tc)
    {
        return VisitLeaf(enode, () => new(0, tc.Value.BytesBuffer.Length));
    }

    private Cost? Visit(ENode enode, Function func)
    {
        return VisitLeaf(enode, () => Cost.Zero);
    }

    private Cost? Visit(ENode enode, Op op)
    {
        return VisitLeaf(enode, () => Cost.Zero);
    }

    private Cost? Visit(ENode enode, TupleConst tc)
    {
        return Visit(enode, costs => costs.Sum());
    }

    private Cost? Visit(ENode enode, IR.Tuple tuple)
    {
        return Visit(enode, costs => costs.Sum());
    }

    private Cost? Visit(ENode enode, Call call)
    {
        return Visit(enode, costs =>
        {
            Cost? cost = null;
            foreach (var targetEnode in _eclasses[enode.Children[0]])
            {
                Cost? newCost;
                if (targetEnode.Expr is Op op)
                {
                    var context = new EGraphOpCostEvaluateContext(call.CheckedType, call.Parameters.Select(x => x.CheckedType).ToArray());
                    newCost = CompilerServices.EvaluateOpCost(op, context);
                }
                else
                {
                    Debug.Assert(targetEnode.Expr is Function);
                    newCost = Visit(targetEnode.Children[0]);
                }

                if (cost == null || (newCost != null && newCost < cost))
                {
                    cost = newCost;
                }
            }

            if (cost == null)
            {
                return null;
            }
            else
            {
                return cost + costs.Sum();
            }
        });
    }

    private Cost VisitLeaf(ENode enode, Func<Cost> costGetter)
    {
        if (!_costs.TryGetValue(enode, out var cost))
        {
            cost = costGetter();
            _changed = true;
        }

        return cost;
    }

    private Cost? Visit(ENode enode, Func<Cost[], Cost?> costGetter)
    {
        var costs = new Cost[enode.Children.Count];
        for (int i = 0; i < costs.Length; i++)
        {
            var childCost = Visit(enode.Children[i]);
            if (childCost != null)
            {
                costs[i] = childCost;
            }
            else
            {
                return null;
            }
        }

        var cost = costGetter(costs);

        if (_costs.TryGetValue(enode, out var oldCost))
        {
            Debug.Assert(cost != null);
            if (oldCost != cost)
            {
                _costs[enode] = cost;
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
