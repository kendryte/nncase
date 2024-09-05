// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using System.Text;
using GiGraph.Dot.Entities.Edges;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Html.Table;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Entities.Subgraphs;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Types.Records;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileTree;

internal sealed class TreePrinter : ITreeNodeVisitor<TreePrinter.Context, TreePrinter.Result>
{
    private readonly Dictionary<OpNode, DotNode> _opMemo;

    private readonly DotSubgraph _subgraph;

    public TreePrinter()
    {
        _opMemo = new();
        Graph = new DotGraph(true);
        _subgraph = new(GiGraph.Dot.Types.Ranks.DotRank.Same);
        Graph.Subgraphs.Add(_subgraph);
    }

    public DotGraph Graph { get; }

    /// <summary>
    /// Get the axis map from current domain to parent domain.
    /// </summary>
    public static Dictionary<int, int> GetDimsMap(ITileAbleNode value)
    {
        var map = new Dictionary<int, int>();
        var relation = value.DomainRelation.Map;
        for (int i = 0; i < relation.Results.Length; i++)
        {
            if (relation.Results[i] is { Offset: AffineDim dim, Extent: AffineExtent ext } && dim.Position == ext.Position)
            {
                map[i] = dim.Position;
            }
        }

        return map;
    }

    public Result Visit(ScopeNode value, Context context)
    {
        var nodes = new List<GiGraph.Dot.Entities.Nodes.DotNode>();
        var relations = new List<DomainRelation>();
        foreach (var child in value.Children)
        {
            var ret = child.Accept(this, context);
            nodes.AddRange(ret.Nodes);
            relations.AddRange(ret.Relations);
        }

        return new(nodes, relations);
    }

    public Result Visit(TileNode value, Context context)
    {
        var (pid, pnames) = context;

        var dimsMap = GetDimsMap(value);
        if (!pnames.Any())
        {
            dimsMap.Clear();
        }

        var strs = new List<string>();
        for (int i = 0; i < value.DimNames.Length; i++)
        {
            var indent = string.Join(string.Empty, Enumerable.Repeat("  ", i));
            var label = dimsMap.ContainsKey(i) ? pnames[dimsMap[i]] : value.DimNames[i];
            strs.Add($"{indent}For {label}");
        }

        var node = Graph.Nodes.Add(value.ToString());
        node.ToRecordNode(rb1 => rb1.AppendField($"Op{value.OpId}").AppendField(string.Join('\n', strs)));

        var result = value.Child.Accept(this, context with { ParentOpId = value.OpId, Names = value.DimNames });

        for (int i = 0; i < result.Nodes.Count; i++)
        {
            Graph.Edges.Add(new DotEdge(node.Id, result.Nodes[i].Id), edge =>
            {
                var r = result.Relations[i];
                edge.Label = $"Op{r.DomainOp} -> Op{r.RangeOp}{System.Environment.NewLine}{r.Map}";
            });
        }

        return new(new() { node }, new() { value.DomainRelation });
    }

    public Result Visit(OpNode value, Context context)
    {
        var (pid, pnames) = context;
        var dimsMap = GetDimsMap(value);

        var node = Graph.Nodes.Add($"{value}");
        _opMemo.Add(value, node);
        _subgraph.Nodes.Add(node);

        node.ToRecordNode(rb => rb.AppendSubrecord(rb1 =>
        {
            rb1.AppendField(value.ToString()).
                AppendField($"{value.Op.GetType().Name}({value.Op.DisplayProperty()})").
                AppendFields(value.DomainBounds.Select((d, i) => $"d{i} : {d}"));
        }).AppendSubrecord(
            rb2 =>
            {
                for (int i = 0; i < value.ReadAccesses.Length; i++)
                {
                    rb2.AppendField($"read {value.ReadAccesses[i]}", $"R{i}");

                    foreach (var dep in value.Dependences.Where(dep => dep.Index == i))
                    {
                        var depNode = _opMemo[dep.Node];
                        Graph.Edges.Add(depNode.Id, node.Id, edge =>
                        {
                            edge.Head.Endpoint.Port = new GiGraph.Dot.Types.Edges.DotEndpointPort($"R{i}");
                            edge.Tail.Endpoint.Port = new GiGraph.Dot.Types.Edges.DotEndpointPort($"W");
                        });
                    }
                }

                rb2.AppendField($"write {value.WriteAccess}", "W");
            }));

        return new(new() { node }, new() { value.DomainRelation });
    }

    internal record Context(int ParentOpId, IReadOnlyList<string> Names)
    {
        public static Context Default => new(-1, Array.Empty<string>());
    }

    internal record Result(List<DotNode> Nodes, List<DomainRelation> Relations)
    {
    }
}
