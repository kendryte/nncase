// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Nncase.CostModel;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Passes;

namespace Nncase.CostModel;

internal sealed class OneShotEGraphCostEvaluator
{
    private EClass? _root;
    private readonly Dictionary<ENode, Cost> _costs = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<EClass, Cost> _eclassCosts = new();
    private readonly HashSet<EClass> _allEclasses = new();

    public ICostEvaluateProvider CostEvaluateProvider { get; }

    public OneShotEGraphCostEvaluator(ICostEvaluateProvider provider)
    {
        CostEvaluateProvider = provider;
    }

    public EGraphCostModel Evaluate(EClass root)
    {
        if (_root is null)
        {
            _root = root;
            PopulateAllEclasses(root);
        }

        var pinner = new ExprPinner(_allEclasses.Select(e => e.Nodes).SelectMany(e => e).Select(e => e.Expr).ToArray());

        foreach (var eclass in _allEclasses)
        {
            foreach (var enode in eclass.Nodes)
            {
                if (!_costs.TryGetValue(enode, out var cost))
                {
                    cost = enode.Expr switch
                    {
                        Call call => Visit(enode, call, eclass.CheckedType),
                        _ => Cost.Zero,
                    };
                    _costs.Add(enode, cost);
                }
            }
        }

        return new(_costs);
    }

    private Cost Visit(ENode enode, Call call, IRType returnType)
    {
        Cost cost = Cost.Zero;
        foreach (var targetEnode in enode.Children[0].Nodes)
        {
            if (targetEnode.Expr is Op op)
            {
                var context = new EGraphCallCostEvaluateContext(returnType, enode.Children.Skip(1).Select(x => x.CheckedType).ToArray(), enode.Children.Skip(1).ToArray());
                cost = CostEvaluateProvider.EvaluateOpCost(op, context);
            }
            else if (targetEnode.Expr is BaseFunction baseFunction)
            {
                var context = new EGraphCallCostEvaluateContext(returnType, enode.Children.Skip(1).Select(x => x.CheckedType).ToArray(), enode.Children.Skip(1).ToArray());
                cost = CostEvaluateProvider.EvaluateBaseFuncCost(baseFunction, context);
            }
        }

        return cost;
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
}

// internal sealed class EGraphCostEvaluator : IEGraphCostEvaluator
// {
//     private EClass _root;
//     private readonly Dictionary<ENode, Cost> _costs = new(ReferenceEqualityComparer.Instance);
//     private readonly Dictionary<EClass, Cost> _eclassCosts = new();
//     private readonly HashSet<EClass> _allEclasses = new();
//     private IBaseFuncCostEvaluator? _baseFuncCostEvaluator;
//     private ICostEvaluator? _costEvaluator;
//     private readonly bool _accumulate;
//     private bool _changed;

//     public EGraphCostEvaluator()
//     {
//         _root = null!;
//         _accumulate = true;
//         _baseFuncCostEvaluator = null;
//         _costEvaluator = null;
//     }

//     public void SetUp(EClass root, IBaseFuncCostEvaluator? basefuncCostEvaluator, ICostEvaluator? costEvaluator)
//     {
//         _root = root;
//         _baseFuncCostEvaluator = basefuncCostEvaluator;
//         _costEvaluator = costEvaluator;
//         PopulateAllEclasses(_root);
//     }

//     public EGraphCostModel Evaluate()
//     {
//         while (true)
//         {
//             _changed = false;
//             TryEvaluateAll();
//             if (!_changed)
//             {
//                 break;
//             }
//         }

//         if (!_eclassCosts.ContainsKey(_root))
//         {
//             throw new InvalidOperationException("Cannot evaluate cost for root.");
//         }

//         return new(_costs);
//     }

//     private void PopulateAllEclasses(EClass eClass)
//     {
//         if (!_allEclasses.Contains(eClass))
//         {
//             _allEclasses.Add(eClass);
//             foreach (var node in eClass.Nodes)
//             {
//                 foreach (var child in node.Children)
//                 {
//                     PopulateAllEclasses(child);
//                 }
//             }
//         }
//     }

//     private void TryEvaluateAll()
//     {
//         foreach (var eclass in _allEclasses)
//         {
//             Visit(eclass);
//         }
//     }

//     private Cost? Visit(EClass eclass)
//     {
//         Cost? cost = null;
//         foreach (var enode in eclass.Nodes)
//         {
//             var newCost = Visit(enode, eclass.CheckedType);
//             if (newCost != null
//                 && (cost == null || newCost < cost))
//             {
//                 cost = newCost;
//             }
//         }

//         return UpdateCost(eclass, cost);
//     }

//     private Cost? Visit(ENode enode, IRType returnType)
//     {
//         return enode.Expr switch
//         {
//             Var var => Visit(enode, var),
//             TensorConst con => Visit(enode, con),
//             TupleConst con => Visit(enode, con),
//             Function func => Visit(enode, func),
//             Call call => Visit(enode, call, returnType),
//             IR.Tuple tuple => Visit(enode, tuple),
//             Op op => Visit(enode, op),
//             If @if => Visit(enode, @if),
//             Marker marker => Visit(enode, marker),
//             None none => Visit(enode, none),
//             BaseFunction baseFunction => Visit(enode, baseFunction),
//             _ => throw new ArgumentException("Unsupported expression type."),
//         };
//     }

//     private Cost Visit(ENode enode, Var var)
//     {
//         return VisitLeaf(enode, () => Cost.Zero);
//     }

//     private Cost Visit(ENode enode, TensorConst tc)
//     {
//         return VisitLeaf(enode, () => Cost.Zero);
//     }

//     private Cost Visit(ENode enode, Op op)
//     {
//         return VisitLeaf(enode, () => Cost.Zero);
//     }

//     private Cost? Visit(ENode enode, Function func)
//     {
//         return Visit(enode, costs => Cost.Zero);
//     }

//     private Cost? Visit(ENode enode, TupleConst tc)
//     {
//         return Visit(enode, costs => costs.Sum());
//     }

//     private Cost? Visit(ENode enode, IR.Tuple tuple)
//     {
//         return Visit(enode, costs => _accumulate ? costs.Sum() : Cost.Zero);
//     }

//     private Cost? Visit(ENode enode, If @if)
//     {
//         return Visit(enode, cost => _accumulate ? cost[^3] + cost[^2] + cost[^1] : Cost.Zero);
//     }

//     private Cost? Visit(ENode enode, Marker marker)
//     {
//         return Visit(enode, costs => _accumulate ? costs[0] : Cost.Zero);
//     }

//     private Cost? Visit(ENode enode, None none)
//     {
//         return Visit(enode, costs => Cost.Zero);
//     }

//     private Cost? Visit(ENode enode, Call call, IRType returnType)
//     {
//         return Visit(enode, costs =>
//         {
//             Cost? cost = null;
//             foreach (var targetEnode in enode.Children[0].Nodes)
//             {
//                 Cost? newCost;
//                 var pinner = new ExprPinner(call);
//                 if (targetEnode.Expr is Op op)
//                 {
//                     var context = new EGraphCallCostEvaluateContext(returnType, enode.Children.Skip(1).Select(x => x.CheckedType).ToArray(), enode.Children.Skip(1).ToArray());
//                     newCost = _costEvaluator?.Visit(context, op) ?? CompilerServices.EvaluateOpCost(op, context);
//                 }
//                 else
//                 {
//                     // Trace.Assert(targetEnode.Expr is Function);
//                     newCost = Visit(targetEnode, returnType);
//                 }

//                 if (cost == null || (newCost != null && newCost < cost))
//                 {
//                     cost = newCost;
//                 }
//             }

//             return UpdateCost(enode, cost == null ? null : (_accumulate ? cost + costs.Sum() : cost));
//         });
//     }

//     private Cost? Visit(ENode enode, BaseFunction baseFunction)
//     {
//         return VisitLeaf(enode, () => _baseFuncCostEvaluator?.Visit(baseFunction) ?? Cost.Zero);
//     }

//     private Cost? UpdateCost(ENode enode, Cost? cost)
//     {
//         if (_costs.TryGetValue(enode, out var oldCost))
//         {
//             if (oldCost != cost)
//             {
//                 if (cost == null)
//                 {
//                     _costs.Remove(enode);
//                 }
//                 else
//                 {
//                     _costs[enode] = cost;
//                 }

//                 _changed = true;
//             }
//         }
//         else
//         {
//             if (cost != null)
//             {
//                 _costs.Add(enode, cost);
//                 _changed = true;
//             }
//         }

//         return cost;
//     }

//     private Cost? UpdateCost(EClass eclass, Cost? cost)
//     {
//         if (_eclassCosts.TryGetValue(eclass, out var oldCost))
//         {
//             if (oldCost != cost)
//             {
//                 if (cost == null)
//                 {
//                     _eclassCosts.Remove(eclass);
//                 }
//                 else
//                 {
//                     _eclassCosts[eclass] = cost;
//                 }

//                 _changed = true;
//             }
//         }
//         else
//         {
//             if (cost != null)
//             {
//                 _eclassCosts.Add(eclass, cost);
//                 _changed = true;
//             }
//         }

//         return cost;
//     }

//     private Cost VisitLeaf(ENode enode, Func<Cost> costGetter)
//     {
//         if (!_costs.TryGetValue(enode, out var cost))
//         {
//             cost = costGetter();
//             _costs[enode] = cost;
//             _changed = true;
//         }

//         return cost;
//     }

//     private Cost? Visit(ENode enode, Func<Cost[], Cost?> costGetter)
//     {
//         var costs = new Cost[enode.Children.Count];
//         for (int i = 0; i < costs.Length; i++)
//         {
//             if (_eclassCosts.TryGetValue(enode.Children[i], out var childCost))
//             {
//                 costs[i] = childCost;
//             }
//             else
//             {
//                 return null;
//             }
//         }

//         var cost = costGetter(costs);
//         return UpdateCost(enode, cost);
//     }
// }

internal sealed class EGraphCallCostEvaluateContext : ICostEvaluateContext
{
    private readonly IRType? _returnType;
    private readonly IRType?[] _argumentTypes;
    private readonly EClass[] _arguments;

    public EGraphCallCostEvaluateContext(IRType? returnType, IRType?[] argumentTypes, EClass[] arguments)
    {
        _returnType = returnType;
        _argumentTypes = argumentTypes;
        _arguments = arguments;
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

    public bool TryGetConstArgument(Op op, ParameterInfo parameter, [MaybeNullWhen(false)] out Const @const)
    {
        var ret = false;
        @const = null;
        foreach (var node in _arguments[parameter.Index].Nodes)
        {
            if (node.Expr is Const c)
            {
                @const = c;
                ret = true;
                break;
            }
        }

        return ret;
    }
}
