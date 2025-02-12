// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Google.OrTools.Sat;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.PatternMatch;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes;

/// <summary>
/// EGraph extract extensions.
/// </summary>
public static class EGraphExtensions
{
    /// <summary>
    /// Extract egraph.
    /// </summary>
    /// <param name="eGraph">egraph.</param>
    /// <param name="root">Root eclass.</param>
    /// <param name="compileOptions">compileOptions.</param>
    /// <param name="basefunc_cost_evaluator">base func cost evaluator.</param>
    /// <param name="constrains">the cp model constrains.</param>
    public static Expr Extract(this IEGraph eGraph, EClass root, CompileOptions compileOptions, Evaluator.IBaseFuncCostEvaluator? basefunc_cost_evaluator, EGraphExtractConstrains[]? constrains = null)
    {
        // 1. set enode expr with more accuracy type.
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
        // var costModel = new CostModel.EGraphCostEvaluator(root.Find(), compileOptions, basefunc_cost_evaluator, false).Evaluate();
        var enodeCostMemo = new Dictionary<ENode, Cost>();
        var opCostMemo = new Dictionary<CostMemoKey, Cost>();
        foreach (var enode in eGraph.Nodes)
        {
            switch (enode.Expr)
            {
                case Call { Target: Expr target } call:
                    switch (target)
                    {
                        case Op op:
                            var returnType = enode.Expr.CheckedType;
                            var key = new CostMemoKey(enode, new CostMemoKeyPartial(op, returnType, enode.Children.Skip(1).Select(x => x.CheckedType).ToArray()));
                            if (!opCostMemo.TryGetValue(key, out var newCost))
                            {
                                var context = new EGraphOpCostEvaluateContext(returnType, enode.Children.Skip(1).Select(x => x.CheckedType).ToArray(), enode.Children.Skip(1).ToArray(), compileOptions);
                                newCost = CompilerServices.EvaluateOpCost(op, context);
                                opCostMemo.Add(key, newCost);
                            }

                            enodeCostMemo[enode] = Cost.Zero;
                            break;
                        default:
                            enodeCostMemo[enode] = Cost.Zero;
                            break;
                    }

                    break;
                default:
                    enodeCostMemo[enode] = Cost.Zero;
                    break;
            }
        }

        var egraphCostModel = new EGraphCostModel(enodeCostMemo);
        return new EGraphExtractor(egraphCostModel).Extract(root.Find(), eGraph, constrains ?? Array.Empty<EGraphExtractConstrains>());
    }
}
