// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Drawing;
using Nncase.Transform;
using Nncase.Pattern;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Entities.Clusters;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Types.Graphs;
using GiGraph.Dot.Types.Nodes;
using GiGraph.Dot.Types.Styling;
using GiGraph.Dot.Types.Colors;
using GiGraph.Dot.Types.Records;
using GiGraph.Dot.Types.Edges;
using System.Linq;

namespace Nncase.Transform
{
    public partial class EGraphPrinter
    {
        public DotGraph AttachEGraphCost(CostModel.EGraphCosts Costs, EClass entry)
        {
            var nodeMap = new Dictionary<ENode, DotNode>();
            foreach (var (eclass, (cost, enode)) in Costs.Context)
            {
                if (OpMaps.ContainsKey(eclass))
                {
                    continue;
                }

                foreach (var dotnode in ClusterMaps[eclass].Nodes.Where((nd => ((DotNode)nd).Id == enode.Expr.GetHashCode().ToString())))
                {
                    nodeMap.Add(enode, (DotNode)dotnode);
                    dotnode.Color = Color.DarkRed;
                }
            }

            dotGraph.Edges.Clear();

            void dfs(EClass curclass)
            {
                var curEnode = Costs.Context[curclass].Item2;
                var curNode = nodeMap[curEnode];
                curNode.Color = Color.RoyalBlue;
                foreach (var (child, i) in curEnode.Children.Select((c, i) => (c, i)))
                {
                    if (OpMaps.ContainsKey(child))
                    {
                        continue;
                    }

                    var paramEnode = Costs[child].Item2;
                    var paramNode = nodeMap[paramEnode];
                    dfs(Costs[paramEnode].Find());
                    dotGraph.Edges.Add(paramNode, curNode, edge =>
                    {
                        edge.Head.Endpoint.Port = new DotEndpointPort($"P{i}");
                        edge.Color = Color.RoyalBlue;
                        edge.Label = Costs[child].Item1.ToString();
                    });
                }
            }

            dfs(entry.Find());
            return dotGraph;
        }

        public static DotGraph DumpEgraphAsDot(EGraph eGraph, CostModel.EGraphCosts Costs, EClass entry, string file)
        {
            var printer = new EGraphPrinter(eGraph);
            printer.ConvertEGraphAsDot();
            printer.AttachEGraphCost(Costs, entry);
            return printer.SaveToFile(file);
        }
    }
}