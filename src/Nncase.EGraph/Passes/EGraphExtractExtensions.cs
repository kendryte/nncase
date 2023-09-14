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

namespace Nncase.Passes;

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
    /// <returns>Extracted root expression.</returns>
    public static Expr Extract(this IEGraph eGraph, EClass root, Evaluator.IBaseFuncCostEvaluator? basefunc_cost_evaluator)
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
        var costModel = new EGraphCostEvaluator(root.Find(), basefunc_cost_evaluator, false).Evaluate();

        // if (DumpScope.Current.IsEnabled(DumpFlags.EGraphCost))
        // {
        //     using var fs = DumpScope.Current.OpenFile(Path.Combine("Costs", $"V{eGraph.Version}.dot"));
        //     EGraphPrinter.DumpEgraphAsDot(eGraph, costModel, root.Find(), fs);
        // }
        // return new EGraphExtractor(costModel).Extract(root.Find(), eGraph);
        return new EGraphExtractors.SatExtractor(costModel).Extract(root.Find(), eGraph);
    }

    /// <summary>
    /// find the minCostEnode in eclass.
    /// <remarks>
    /// the marker first.
    /// </remarks>
    /// </summary>
    internal static ENode MinByWithMarker(this EClass eClass, CostModel.EGraphCostModel costModel)
    {
        return eClass.Nodes.OrderBy(e => e.Expr, ENodeTypeComparer.Instance).MinBy(x => x.Expr is Marker ? Cost.Zero : costModel[x])!;
    }

    /// <summary>
    /// find the minCostEnode in eclass skip marker.
    /// </summary>
    internal static ENode MinByWithOutMarker(this EClass eClass, CostModel.EGraphCostModel costModel)
    {
        return eClass.Nodes.Where(e => e.Expr is not Marker).MinBy(x => costModel[x])!;
    }

    internal sealed class ENodeTypeComparer : IComparer<Expr>
    {
        public static readonly ENodeTypeComparer Instance = new();

        public int Compare(Expr? x, Expr? y) => (x, y) switch
        {
            (null, null) => 0,
            (Expr, null) => 1,
            (null, Expr) => -1,
            (Expr, Expr) => GetPriority(x).CompareTo(GetPriority(y)),
        };

        private int GetPriority(Expr x) => x switch
        {
            Marker => 0,
            Const => 1,
            _ => 2,
        };
    }
}
