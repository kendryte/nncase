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

                var minCostEnode = parent.MinByWithMarker(costModel);

                // when this marker ecalss has been visited, skip it.
                if (markerEclassMemo.Contains(parent))
                {
                    minCostEnode = parent.MinByWithOutMarker(costModel);
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
                        var otherminCostENode = child.MinByWithOutMarker(costModel);
                        var (childDotNode, _) = NodesMap[otherminCostENode];
                        _dotGraph.Edges.Add(childDotNode, minCostDotnode, edge =>
                        {
                            edge.Head.Endpoint.Port = new DotEndpointPort($"P{i}");
                            edge.Color = Color.SpringGreen;
                        });
                    }
                    else
                    {
                        var childEnode = child.Find().MinByWithMarker(costModel);
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
