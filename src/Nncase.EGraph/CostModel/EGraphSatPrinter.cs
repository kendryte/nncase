﻿// Copyright (c) Canaan Inc. All rights reserved.
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
    internal static DotGraph DumpEgraphAsDot(IEGraph eGraph, CostModel.EGraphCostModel costModel, IReadOnlyDictionary<ENode, bool> pick, EClass entry, Stream file)
    {
        var printer = new EGraphPrinter(eGraph);
        printer.ConvertEGraphAsDot();
        printer.AttachEGraphCostPick(costModel, pick);
        return printer.SaveToStream(file);
    }

    private DotGraph AttachEGraphCostPick(CostModel.EGraphCostModel costModel, IReadOnlyDictionary<ENode, bool> pick)
    {
        // 1. display each enode costs.
        foreach (var (enode, (dotnode, table)) in NodesMap)
        {
            var cost = costModel[enode];
            if (cost != CostModel.Cost.Zero)
            {
                table.AddRow(row =>
                {
                    foreach (var (k, v) in cost.Factors)
                    {
                        row.AddCell($"{k}: {v:F2}");
                    }

                    row.AddCell($"Score: {cost.Score:F2}");
                });
            }

            dotnode.ToPlainHtmlNode(table);
        }

        foreach (var (enode, picked) in pick)
        {
            if (picked && NodesMap.TryGetValue(enode, out var p))
            {
                p.Node.Color = Color.DeepSkyBlue;
            }
        }

        return _dotGraph;
    }
}
