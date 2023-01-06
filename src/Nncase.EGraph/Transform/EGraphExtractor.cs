// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Transform;

/// <summary>
/// EGraph extract extensions.
/// </summary>
public static class EGraphExtractExtensions
{
    /// <summary>
    /// Extract egraph.
    /// </summary>
    /// <param name="eGraph">eGraph.</param>
    /// <param name="root">Root eclass.</param>
    /// <param name="basefunc_cost_evaluator">base func cost evaluator.</param>
    /// <param name="options">Options.</param>
    /// <returns>Extracted root expression.</returns>
    public static Expr Extract(this EGraph eGraph, EClass root, Evaluator.IBaseFuncCostEvaluator? basefunc_cost_evaluator, RunPassOptions options)
    {
        // 1. set the all expr checked shape
        foreach (var eclass in eGraph.Classes)
        {
            foreach (var nodes in eclass.Nodes)
            {
                if (eclass.CheckedType.CompareTo(nodes.Expr.CheckedType) > 0)
                {
                    nodes.Expr.CheckedType = eclass.CheckedType;
                }
            }
        }

        // 2. start the cost evaluator
        var costModel = new EGraphCostEvaluator(root.Find(), basefunc_cost_evaluator).Evaluate();
        if (options.DumpLevel > 2)
        {
            EGraphPrinter.DumpEgraphAsDot(eGraph, costModel, root.Find(), Path.Combine(options.DumpDir, "Costs", $"V{eGraph.Version}"));
        }

        return new EGraphExtractor(costModel).Extract(root.Find());
    }
}

internal class EGraphExtractor
{
    private readonly EGraphCostModel _costModel;
    private readonly Dictionary<EClass, Expr> _eclassMemo = new();
    private readonly Dictionary<EClass, Expr> _markerEclassMemo = new();

    public EGraphExtractor(EGraphCostModel costModel)
    {
        _costModel = costModel;
    }

    public Expr Extract(EClass root)
    {
        Visit(root);
        return _eclassMemo[root];
    }

    private void Visit(EClass eclass)
    {
        var stack = new Stack<(EClass, ENode)>();
        stack.Push((eclass, eclass.Nodes.MinBy(x => _costModel[x])!));
        var markerEclassSet = new HashSet<EClass>();
        while (stack.Any())
        {
            (eclass, var minCostEnode) = stack.Peek();
            if (_eclassMemo.TryGetValue(eclass, out var expr))
            {
                stack.Pop();
                continue;
            }

            switch (minCostEnode.Expr)
            {
                case Var or TensorConst or TupleConst or Op or Fusion or None:
                    expr = minCostEnode.Expr;
                    _eclassMemo.Add(eclass, expr);
                    stack.Pop();
                    break;
                case Function or Call or IR.Tuple or Marker:
                    var childrenExprs = new List<Expr>();
                    foreach (var child in minCostEnode.Children)
                    {
                        if (!_eclassMemo.TryGetValue(child, out var childExpr))
                        {
                            if (minCostEnode.Expr is Marker && child == eclass)
                            {
                                if (!_markerEclassMemo.TryGetValue(eclass, out var markerInputExpr))
                                {
                                    markerEclassSet.Add(eclass);
                                    stack.Push((eclass, eclass.Nodes.Where(n => n.Expr is not Marker).MinBy(x => _costModel[x])!));
                                }
                                else
                                {
                                    childrenExprs.Add(markerInputExpr);
                                }
                            }
                            else
                            {
                                stack.Push((child, child.Nodes.MinBy(x => _costModel[x])!));
                            }
                        }
                        else
                        {
                            childrenExprs.Add(childExpr);
                        }
                    }

                    if (childrenExprs.Count != minCostEnode.Children.Count)
                    {
                        break;
                    }

                    expr = minCostEnode.Expr switch
                    {
                        Function function => Visit(minCostEnode, function, new(childrenExprs)),
                        Call call => Visit(minCostEnode, call, new(childrenExprs)),
                        IR.Tuple tuple => Visit(minCostEnode, tuple, new(childrenExprs)),
                        Marker marker => Visit(minCostEnode, marker, new(childrenExprs)),
                        _ => throw new ArgumentException("Unsupported expression type."),
                    };
                    if (markerEclassSet.Contains(eclass) && minCostEnode.Expr is not Marker)
                    {
                        _markerEclassMemo.Add(eclass, expr);
                    }
                    else
                    {
                        _eclassMemo.Add(eclass, expr);
                    }

                    stack.Pop();
                    break;
                default:
                    throw new ArgumentException("Unsupported expression type.");
            }
        }

        // var callPattern = IsCall(IsWildcard(), IsWildcard());
        // var isCallExpr = callPattern.MatchLeaf(expr);
        // if (isCallExpr == true)
        // {
        //     if (((Call)expr).EnodeQuantConfigWithCosine != null)
        //     {
        //         var pattern = IsCall(IsWildcard(), IsWildcard());
        //         var isCall = pattern.MatchLeaf(expr);
        //         if (isCall == true)
        //         {
        //             System.Console.WriteLine(expr + "  " + expr.CheckedType);
        //             for (int i = 0; i < ((Call)expr).EnodeQuantConfigWithCosine.Count; i++)
        //             {
        //                 for (int j = 0; j < ((Call)expr).EnodeQuantConfigWithCosine[i].Item1.Count; j++)
        //                 {
        //                     System.Console.Write(((Call)expr).EnodeQuantConfigWithCosine[i].Item1[j] + "  ");
        //                 }

        // System.Console.WriteLine(((Call)expr).EnodeQuantConfigWithCosine[i].Item3);
        //             }
        //         }
        //     }
        // }

        // return expr;
    }

    private Marker Visit(ENode enode, Marker marker, IRArray<Expr> children)
    {
        var target = children[0];
        var attr = children[1];
        return marker with { Target = target, Attribute = attr };
    }

    private Function Visit(ENode enode, Function func, IRArray<Expr> children)
    {
        var body = children[0];
        return func with { Body = body };
    }

    private IR.Tuple Visit(ENode enode, IR.Tuple tuple, IRArray<Expr> children)
    {
        return tuple with { Fields = children };
    }

    private Call Visit(ENode enode, Call call, IRArray<Expr> children)
    {
        var target = children[0];
        var parameters = children.Skip(1);
        return call with { Target = target, Parameters = new(parameters) };
    }
}
