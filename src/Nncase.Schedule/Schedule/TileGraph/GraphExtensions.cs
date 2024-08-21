// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Text.RegularExpressions;
using Google.OrTools.ConstraintSolver;
using QuikGraph.Graphviz;

namespace Nncase.Schedule.TileGraph;

public static class GraphExtensions
{
    private static readonly QuikGraph.Graphviz.Dot.GraphvizColor[] _colors = new[] { QuikGraph.Graphviz.Dot.GraphvizColor.LightBlue, QuikGraph.Graphviz.Dot.GraphvizColor.LightCoral, QuikGraph.Graphviz.Dot.GraphvizColor.LightGreen, QuikGraph.Graphviz.Dot.GraphvizColor.LightPink, QuikGraph.Graphviz.Dot.GraphvizColor.LightCyan, QuikGraph.Graphviz.Dot.GraphvizColor.LightSalmon, QuikGraph.Graphviz.Dot.GraphvizColor.LightGoldenrodYellow, QuikGraph.Graphviz.Dot.GraphvizColor.LightGray, QuikGraph.Graphviz.Dot.GraphvizColor.LightSeaGreen, QuikGraph.Graphviz.Dot.GraphvizColor.LightSkyBlue, QuikGraph.Graphviz.Dot.GraphvizColor.LightSlateGray, QuikGraph.Graphviz.Dot.GraphvizColor.LightYellow, QuikGraph.Graphviz.Dot.GraphvizColor.LightSteelBlue, QuikGraph.Graphviz.Dot.GraphvizColor.AliceBlue, QuikGraph.Graphviz.Dot.GraphvizColor.AntiqueWhite, QuikGraph.Graphviz.Dot.GraphvizColor.Aqua, QuikGraph.Graphviz.Dot.GraphvizColor.Aquamarine, QuikGraph.Graphviz.Dot.GraphvizColor.Azure, QuikGraph.Graphviz.Dot.GraphvizColor.Beige, QuikGraph.Graphviz.Dot.GraphvizColor.Bisque, QuikGraph.Graphviz.Dot.GraphvizColor.Black, QuikGraph.Graphviz.Dot.GraphvizColor.BlanchedAlmond, QuikGraph.Graphviz.Dot.GraphvizColor.Blue, QuikGraph.Graphviz.Dot.GraphvizColor.BlueViolet, QuikGraph.Graphviz.Dot.GraphvizColor.Brown, QuikGraph.Graphviz.Dot.GraphvizColor.BurlyWood, QuikGraph.Graphviz.Dot.GraphvizColor.CadetBlue, QuikGraph.Graphviz.Dot.GraphvizColor.Chartreuse, QuikGraph.Graphviz.Dot.GraphvizColor.Chocolate, QuikGraph.Graphviz.Dot.GraphvizColor.Coral, QuikGraph.Graphviz.Dot.GraphvizColor.CornflowerBlue, QuikGraph.Graphviz.Dot.GraphvizColor.Cornsilk, QuikGraph.Graphviz.Dot.GraphvizColor.Crimson, QuikGraph.Graphviz.Dot.GraphvizColor.Cyan, QuikGraph.Graphviz.Dot.GraphvizColor.DarkBlue, QuikGraph.Graphviz.Dot.GraphvizColor.DarkCyan, QuikGraph.Graphviz.Dot.GraphvizColor.DarkGoldenrod, QuikGraph.Graphviz.Dot.GraphvizColor.DarkGray, QuikGraph.Graphviz.Dot.GraphvizColor.DarkGreen, QuikGraph.Graphviz.Dot.GraphvizColor.DarkKhaki, QuikGraph.Graphviz.Dot.GraphvizColor.DarkMagenta, QuikGraph.Graphviz.Dot.GraphvizColor.DarkOliveGreen, QuikGraph.Graphviz.Dot.GraphvizColor.DarkOrange, QuikGraph.Graphviz.Dot.GraphvizColor.DarkOrchid, QuikGraph.Graphviz.Dot.GraphvizColor.DarkRed, QuikGraph.Graphviz.Dot.GraphvizColor.DarkSalmon, QuikGraph.Graphviz.Dot.GraphvizColor.DarkSeaGreen, QuikGraph.Graphviz.Dot.GraphvizColor.DarkSlateBlue, QuikGraph.Graphviz.Dot.GraphvizColor.DarkSlateGray, QuikGraph.Graphviz.Dot.GraphvizColor.DarkTurquoise, QuikGraph.Graphviz.Dot.GraphvizColor.DarkViolet, QuikGraph.Graphviz.Dot.GraphvizColor.DeepPink, QuikGraph.Graphviz.Dot.GraphvizColor.DeepSkyBlue, QuikGraph.Graphviz.Dot.GraphvizColor.DimGray, QuikGraph.Graphviz.Dot.GraphvizColor.DodgerBlue, QuikGraph.Graphviz.Dot.GraphvizColor.Firebrick, QuikGraph.Graphviz.Dot.GraphvizColor.FloralWhite, QuikGraph.Graphviz.Dot.GraphvizColor.ForestGreen, QuikGraph.Graphviz.Dot.GraphvizColor.Fuchsia, QuikGraph.Graphviz.Dot.GraphvizColor.Gainsboro, QuikGraph.Graphviz.Dot.GraphvizColor.GhostWhite, QuikGraph.Graphviz.Dot.GraphvizColor.Gold, QuikGraph.Graphviz.Dot.GraphvizColor.Goldenrod, QuikGraph.Graphviz.Dot.GraphvizColor.Gray, QuikGraph.Graphviz.Dot.GraphvizColor.Green, QuikGraph.Graphviz.Dot.GraphvizColor.GreenYellow, QuikGraph.Graphviz.Dot.GraphvizColor.Honeydew, QuikGraph.Graphviz.Dot.GraphvizColor.HotPink, QuikGraph.Graphviz.Dot.GraphvizColor.IndianRed, QuikGraph.Graphviz.Dot.GraphvizColor.Indigo, QuikGraph.Graphviz.Dot.GraphvizColor.Ivory, QuikGraph.Graphviz.Dot.GraphvizColor.Khaki, QuikGraph.Graphviz.Dot.GraphvizColor.Lavender, QuikGraph.Graphviz.Dot.GraphvizColor.LavenderBlush, QuikGraph.Graphviz.Dot.GraphvizColor.LawnGreen, QuikGraph.Graphviz.Dot.GraphvizColor.LemonChiffon, QuikGraph.Graphviz.Dot.GraphvizColor.Lime, QuikGraph.Graphviz.Dot.GraphvizColor.LimeGreen, QuikGraph.Graphviz.Dot.GraphvizColor.Linen, QuikGraph.Graphviz.Dot.GraphvizColor.Magenta, QuikGraph.Graphviz.Dot.GraphvizColor.Maroon, QuikGraph.Graphviz.Dot.GraphvizColor.MediumAquamarine, QuikGraph.Graphviz.Dot.GraphvizColor.MediumBlue, QuikGraph.Graphviz.Dot.GraphvizColor.MediumOrchid, QuikGraph.Graphviz.Dot.GraphvizColor.MediumPurple, QuikGraph.Graphviz.Dot.GraphvizColor.MediumSeaGreen, QuikGraph.Graphviz.Dot.GraphvizColor.MediumSlateBlue, QuikGraph.Graphviz.Dot.GraphvizColor.MediumSpringGreen, QuikGraph.Graphviz.Dot.GraphvizColor.MediumTurquoise, QuikGraph.Graphviz.Dot.GraphvizColor.MediumVioletRed, QuikGraph.Graphviz.Dot.GraphvizColor.MidnightBlue, QuikGraph.Graphviz.Dot.GraphvizColor.MintCream, QuikGraph.Graphviz.Dot.GraphvizColor.MistyRose, QuikGraph.Graphviz.Dot.GraphvizColor.Moccasin, QuikGraph.Graphviz.Dot.GraphvizColor.NavajoWhite, QuikGraph.Graphviz.Dot.GraphvizColor.Navy, QuikGraph.Graphviz.Dot.GraphvizColor.OldLace, QuikGraph.Graphviz.Dot.GraphvizColor.Olive, QuikGraph.Graphviz.Dot.GraphvizColor.OliveDrab, QuikGraph.Graphviz.Dot.GraphvizColor.Orange, QuikGraph.Graphviz.Dot.GraphvizColor.OrangeRed, QuikGraph.Graphviz.Dot.GraphvizColor.Orchid, QuikGraph.Graphviz.Dot.GraphvizColor.PaleGoldenrod, QuikGraph.Graphviz.Dot.GraphvizColor.PaleGreen, QuikGraph.Graphviz.Dot.GraphvizColor.PaleTurquoise, QuikGraph.Graphviz.Dot.GraphvizColor.PaleVioletRed, QuikGraph.Graphviz.Dot.GraphvizColor.PapayaWhip, QuikGraph.Graphviz.Dot.GraphvizColor.PeachPuff, QuikGraph.Graphviz.Dot.GraphvizColor.Peru, QuikGraph.Graphviz.Dot.GraphvizColor.Pink, QuikGraph.Graphviz.Dot.GraphvizColor.Plum, QuikGraph.Graphviz.Dot.GraphvizColor.PowderBlue, QuikGraph.Graphviz.Dot.GraphvizColor.Purple, QuikGraph.Graphviz.Dot.GraphvizColor.Red, QuikGraph.Graphviz.Dot.GraphvizColor.RosyBrown, QuikGraph.Graphviz.Dot.GraphvizColor.RoyalBlue, QuikGraph.Graphviz.Dot.GraphvizColor.SaddleBrown, QuikGraph.Graphviz.Dot.GraphvizColor.Salmon, QuikGraph.Graphviz.Dot.GraphvizColor.SandyBrown, QuikGraph.Graphviz.Dot.GraphvizColor.SeaGreen, QuikGraph.Graphviz.Dot.GraphvizColor.SeaShell, QuikGraph.Graphviz.Dot.GraphvizColor.Sienna, QuikGraph.Graphviz.Dot.GraphvizColor.Silver, QuikGraph.Graphviz.Dot.GraphvizColor.SkyBlue, QuikGraph.Graphviz.Dot.GraphvizColor.SlateBlue, QuikGraph.Graphviz.Dot.GraphvizColor.SlateGray, QuikGraph.Graphviz.Dot.GraphvizColor.Snow, QuikGraph.Graphviz.Dot.GraphvizColor.SpringGreen, QuikGraph.Graphviz.Dot.GraphvizColor.SteelBlue, QuikGraph.Graphviz.Dot.GraphvizColor.Tan, QuikGraph.Graphviz.Dot.GraphvizColor.Teal, QuikGraph.Graphviz.Dot.GraphvizColor.Thistle, QuikGraph.Graphviz.Dot.GraphvizColor.Tomato, QuikGraph.Graphviz.Dot.GraphvizColor.Transparent, QuikGraph.Graphviz.Dot.GraphvizColor.Turquoise, QuikGraph.Graphviz.Dot.GraphvizColor.Violet, QuikGraph.Graphviz.Dot.GraphvizColor.Wheat, QuikGraph.Graphviz.Dot.GraphvizColor.White, QuikGraph.Graphviz.Dot.GraphvizColor.WhiteSmoke, QuikGraph.Graphviz.Dot.GraphvizColor.Yellow, QuikGraph.Graphviz.Dot.GraphvizColor.YellowGreen };

    public static string ToGraphViz(this TileGraph graph)
    {
        return graph.ToGraphviz(alg =>
            {
                alg.FormatCluster += (_, arg) =>
                {
                    if (arg.Cluster is TileGraph tg)
                    {
                        var name = $"Op{tg.OpId}@L{tg.Level}";
                        arg.GraphFormat.Label = name;
                        if (tg.DomainRelation is not null)
                        {
                            arg.GraphFormat.Label += System.Environment.NewLine + tg.DomainRelation.ToString();
                        }

                        arg.GraphFormat.LabelLocation = QuikGraph.Graphviz.Dot.GraphvizLabelLocation.T;
                        arg.GraphFormat.LabelJustification = QuikGraph.Graphviz.Dot.GraphvizLabelJustification.L;
                        arg.GraphFormat.BackgroundColor = _colors[tg.OpId];
                    }
                };

                alg.FormatVertex += (_, arg) =>
                {
                    var cell = new QuikGraph.Graphviz.Dot.GraphvizRecordCell();
                    cell.Cells.Add(new() { Text = arg.Vertex.ToString(), Port = "Title" });
                    cell.Cells.Add(new() { Text = arg.Vertex.DomainRelation.ToString() });
                    for (int i = 0; i < arg.Vertex.ReadAccesses.Length; i++)
                    {
                        var item = arg.Vertex.ReadAccesses[i];
                        cell.Cells.Add(new() { Text = $"read: {item}", Port = $"R{i}" });
                    }

                    cell.Cells.Add(new() { Text = $"write: {arg.Vertex.WriteAccess}", Port = $"W" });

                    arg.VertexFormat.Record.Cells.Add(cell);
                    arg.VertexFormat.Shape = QuikGraph.Graphviz.Dot.GraphvizVertexShape.Record;
                    arg.VertexFormat.FillColor = _colors[arg.Vertex.OpId];
                    arg.VertexFormat.Style = QuikGraph.Graphviz.Dot.GraphvizVertexStyle.Filled;
                };

                alg.FormatEdge += (_, arg) =>
                {
                    arg.EdgeFormat.TailPort = $"W:e";
                    arg.EdgeFormat.HeadPort = $"R{arg.Edge.Index}:e";
                };
            });
    }

    public static void Dump(this TileGraph graph, string name)
    {
        using (var file = Diagnostics.DumpScope.Current.OpenFile($"{name}.dot"))
        {
            using var writer = new StreamWriter(file);
            writer.Write(ToGraphViz(graph));
        }
    }

    public static bool Merge(this TileGraph graph, MergePoint mergePoint)
    {
        var merger = new GraphMerger(mergePoint.Consumer, mergePoint.Producer, mergePoint.Level);
        return merger.Visit(graph);
    }

    public static void Walk(this TileGraph graph, Action<TileGraph> func, bool postOrder = false)
    {
        if (!postOrder)
        {
            func(graph);
        }

        foreach (var subgraph in graph.Clusters.OfType<TileGraph>())
        {
            Walk(subgraph, func);
        }

        if (postOrder)
        {
            func(graph);
        }
    }

    public static TileGraph? GetParentTileableNode(this ITileableNode node, TileGraph rootGraph)
    {
        TileGraph? parent = null;
        if (node is TileGraph graph)
        {
            parent = (TileGraph)graph.Parent!;
        }
        else if (node is OpNode)
        {
            rootGraph.Walk(s =>
            {
                if (s.Level == 1 && s.Vertices.Contains(node))
                {
                    parent = s;
                }
            });
        }

        return parent;
    }

    public static TileGraph RootParent(this TileGraph graph)
    {
        var current = graph;
        while (current.Parent is TileGraph parent)
        {
            current = parent;
        }

        return current;
    }

    public static TileGraph Clone(this TileGraph sourceGraph)
    {
        if (sourceGraph.Parent is not null)
        {
            throw new NotSupportedException("can't clone non root graph");
        }

        var wrappedGraph = new QuikGraph.AdjacencyGraph<OpNode, OpEdge>();
        var targetGraph = new TileGraph(sourceGraph.Level, wrappedGraph);
        CloneInternal(sourceGraph, targetGraph);

        foreach (var item in sourceGraph.Edges)
        {
            wrappedGraph.AddEdge(item);
        }

        return targetGraph;
    }

    public static IR.Expr GetArgument(this IR.Affine.Grid grid, int index)
    {
        return index >= grid.Reads.Length ? grid.Buffers[^1] : grid.Reads[index];
    }

    private static void CloneInternal(TileGraph sourceGraph, TileGraph destGraph)
    {
        if (sourceGraph.ClustersCount != 0)
        {
            foreach (var sourceChild in sourceGraph.Clusters.OfType<TileGraph>())
            {
                var destChild = destGraph.CreateCluster<TileGraph>(sourceChild.Level, sourceChild.OpId, sourceChild.DomainRelation);
                CloneInternal(sourceChild, destChild);
            }
        }
        else
        {
            foreach (var item in sourceGraph.Vertices)
            {
                destGraph.AddVertex(item);
            }
        }
    }
}
