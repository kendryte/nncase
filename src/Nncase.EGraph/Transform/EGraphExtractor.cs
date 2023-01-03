// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Nncase.CostModel;
using Nncase.Diagnostics;
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
    /// <param name="context">Context.</param>
    /// <returns>Extracted root expression.</returns>
    public static Expr Extract(this EGraph eGraph, EClass root, Evaluator.IBaseFuncCostEvaluator? basefunc_cost_evaluator, RunPassContext context)
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
        if (context.Dumpper.IsEnabled(DumpFlags.EGraphCost))
        {
            using var fs = context.Dumpper.OpenWrite(Path.Combine("Costs", $"V{eGraph.Version}"));
            EGraphPrinter.DumpEgraphAsDot(eGraph, costModel, root.Find(), fs);
        }

        return new EGraphExtractor(costModel).Extract(root.Find());
    }
}

internal class EGraphExtractor
{
    private readonly EGraphCostModel _costModel;
    private readonly Dictionary<EClass, Expr> _eclassMemo = new();

    public EGraphExtractor(EGraphCostModel costModel)
    {
        _costModel = costModel;
    }

    public Expr Extract(EClass root)
    {
        return Visit(root);
    }

    private Expr Visit(EClass eclass)
    {
        if (!_eclassMemo.TryGetValue(eclass, out var expr))
        {
            var minCostEnode = eclass.Nodes.MinBy(x => _costModel[x]);
            expr = minCostEnode!.Expr switch
            {
                Var var => VisitLeaf(minCostEnode, var),
                TensorConst con => VisitLeaf(minCostEnode, con),
                TupleConst con => VisitLeaf(minCostEnode, con),
                Function func => Visit(minCostEnode, func),
                Call call => Visit(minCostEnode, call),
                IR.Tuple tuple => Visit(minCostEnode, tuple),
                Op op => VisitLeaf(minCostEnode, op),
                Marker marker => Visit(minCostEnode, marker),
                None none => Visit(minCostEnode, none),
                Fusion fusion => VisitLeaf(minCostEnode, fusion),
                _ => throw new ArgumentException("Unsupported expression type."),
            };
            _eclassMemo.Add(eclass, expr);
        }

        var callPattern = IsCall(IsWildcard(), IsWildcard());
        var isCallExpr = callPattern.MatchLeaf(expr);
        if (isCallExpr == true)
        {
            if (((Call)expr).EnodeQuantConfigWithCosine != null)
            {
                var pattern = IsCall(IsWildcard(), IsWildcard());
                var isCall = pattern.MatchLeaf(expr);
                if (isCall == true)
                {
                    System.Console.WriteLine(expr + "  " + expr.CheckedType);
                    for (int i = 0; i < ((Call)expr).EnodeQuantConfigWithCosine.Count; i++)
                    {
                        for (int j = 0; j < ((Call)expr).EnodeQuantConfigWithCosine[i].Item1.Count; j++)
                        {
                            System.Console.Write(((Call)expr).EnodeQuantConfigWithCosine[i].Item1[j] + "  ");
                        }

                        System.Console.WriteLine(((Call)expr).EnodeQuantConfigWithCosine[i].Item3);
                    }
                }
            }
        }

        return expr;
    }

    private Expr VisitLeaf(ENode enode, Expr expr)
    {
        return expr;
    }

    private Marker Visit(ENode enode, Marker marker)
    {
        var target = Visit(enode.Children[0]);
        var attr = Visit(enode.Children[1]);
        return marker with { Target = target, Attribute = attr };
    }

    private None Visit(ENode enode, None none)
    {
        return none;
    }

    private Function Visit(ENode enode, Function func)
    {
        var body = Visit(enode.Children[0]);
        return func with { Body = body };
    }

    private IR.Tuple Visit(ENode enode, IR.Tuple tuple)
    {
        var fields = enode.Children.Select(Visit);
        return tuple with { Fields = new(fields) };
    }

    private Call Visit(ENode enode, Call call)
    {
        var target = Visit(enode.Children[0]);
        var parameters = enode.Children.Skip(1).Select(Visit);
        return call with { Target = target, Parameters = new(parameters) };
    }
}
