// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using GiGraph.Dot.Entities.Clusters;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Types.Colors;
using GiGraph.Dot.Types.Edges;
using GiGraph.Dot.Types.Graphs;
using GiGraph.Dot.Types.Nodes;
using GiGraph.Dot.Types.Records;
using GiGraph.Dot.Types.Styling;
using Nncase.IR;
using Nncase.Passes;
using Nncase.PatternMatch;

namespace Nncase.Passes;

public partial class EGraphPrinter
{
    internal static DotGraph DumpEgraphAsDot(IEGraph eGraph, CostModel.EGraphCostModel costModel, EClass entry, Stream file)
    {
        var printer = new EGraphPrinter(eGraph);
        printer.ConvertEGraphAsDot();
        printer.AttachEGraphCost(costModel, entry);
        return printer.SaveToStream(file);
    }

    /// <summary>
    /// find the minCostEnode in eclass.
    /// <remarks>
    /// the marker first.
    /// </remarks>
    /// </summary>
    internal static ENode MinByWithMarker(EClass eClass, CostModel.EGraphCostModel costModel)
    {
        return eClass.Nodes.OrderBy(e => e.Expr, ENodeTypeComparer.Instance).MinBy(x => x.Expr is Marker ? CostModel.Cost.Zero : costModel[x])!;
    }

    /// <summary>
    /// find the minCostEnode in eclass skip marker.
    /// </summary>
    internal static ENode MinByWithOutMarker(EClass eClass, CostModel.EGraphCostModel costModel)
    {
        return eClass.Nodes.Where(e => e.Expr is not Marker).MinBy(x => costModel[x])!;
    }

    private DotGraph AttachEGraphCost(CostModel.EGraphCostModel costModel, EClass entry)
    {
        // 1. display each enode costs.
        foreach (var (enode, (dotnode, table)) in NodesMap)
        {
            if (enode.Expr is IR.Var or IR.Op or IR.None)
            {
                continue;
            }

            table.AddRow(row =>
            {
                var cost = costModel[enode];
                foreach (var (k, v) in cost.Factors)
                {
                    row.AddCell($"{k}: {v:F2}");
                }

                row.AddCell($"Score: {cost.Score:F2}");
            });
            dotnode.ToPlainHtmlNode(table);
        }

        _dotGraph.Edges.Clear();

        HashSet<EClass> eclassMemo = new();
        HashSet<EClass> markerEclassMemo = new();

        void Dfs(EClass curclass)
        {
            var stack = new Stack<EClass>();
            stack.Push(curclass);
            while (stack.Any())
            {
                var parent = stack.Pop();
                if (eclassMemo.Contains(parent) || _opMaps.ContainsKey(parent))
                {
                    continue;
                }

                var minCostEnode = MinByWithMarker(parent, costModel);

                // when this marker ecalss has been visited, skip it.
                if (markerEclassMemo.Contains(parent))
                {
                    minCostEnode = MinByWithOutMarker(parent, costModel);
                }

                var (minCostDotnode, table) = NodesMap[minCostEnode];
                minCostDotnode.Color = Color.DeepSkyBlue;
                foreach (var (child, i) in minCostEnode.Children.Select((c, i) => (c, i)))
                {
                    if (_opMaps.ContainsKey(child))
                    {
                        continue;
                    }

                    // note when marker child is it's self need select other node.
                    if (minCostEnode.Expr is Marker && child == parent)
                    {
                        markerEclassMemo.Add(child);
                        var otherminCostENode = MinByWithOutMarker(child, costModel);
                        var (childDotNode, _) = NodesMap[otherminCostENode];
                        _dotGraph.Edges.Add(childDotNode, minCostDotnode, edge =>
                        {
                            edge.Head.Endpoint.Port = new DotEndpointPort($"P{i}");
                            edge.Color = Color.SpringGreen;
                        });
                    }
                    else
                    {
                        var childEnode = MinByWithMarker(child.Find(), costModel);
                        var (childDotNode, _) = NodesMap[childEnode];
                        _dotGraph.Edges.Add(childDotNode, minCostDotnode, edge =>
                        {
                            edge.Head.Endpoint.Port = new DotEndpointPort($"P{i}");
                            edge.Color = Color.SpringGreen;
                        });
                    }

                    stack.Push(child);
                }

                if (!markerEclassMemo.Contains(parent))
                {
                    eclassMemo.Add(parent);
                }
            }
        }

        Dfs(entry.Find());
        return _dotGraph;
    }
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
