// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
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
using Nncase.Passes;
using Nncase.PatternMatch;

namespace Nncase.Passes;

public partial class EGraphPrinter
{
    private readonly Color[] _knownColors = new Color[] { Color.MediumAquamarine, Color.MediumBlue, Color.MediumOrchid, Color.MediumPurple, Color.MediumSeaGreen, Color.MediumSlateBlue, Color.MediumSpringGreen, Color.Maroon, Color.MediumTurquoise, Color.MidnightBlue, Color.MintCream, Color.MistyRose, Color.Moccasin, Color.NavajoWhite, Color.Navy, Color.OldLace, Color.MediumVioletRed, Color.Magenta, Color.Linen, Color.LimeGreen, Color.LavenderBlush, Color.LawnGreen, Color.LemonChiffon, Color.LightBlue, Color.LightCoral, Color.LightCyan, Color.LightGoldenrodYellow, Color.LightGray, Color.LightGreen, Color.LightPink, Color.LightSalmon, Color.LightSeaGreen, Color.LightSkyBlue, Color.LightSlateGray, Color.LightSteelBlue, Color.LightYellow, Color.Lime, Color.Olive, Color.OliveDrab, Color.Orange, Color.OrangeRed, Color.Silver, Color.SkyBlue, Color.SlateBlue, Color.SlateGray, Color.Snow, Color.SpringGreen, Color.SteelBlue, Color.Tan, Color.Teal, Color.Thistle, Color.Tomato, Color.Transparent, Color.Turquoise, Color.Violet, Color.Wheat, Color.White, Color.WhiteSmoke, Color.Sienna, Color.Lavender, Color.SeaShell, Color.SandyBrown, Color.Orchid, Color.PaleGoldenrod, Color.PaleGreen, Color.PaleTurquoise, Color.PaleVioletRed, Color.PapayaWhip, Color.PeachPuff, Color.Peru, Color.Pink, Color.Plum, Color.PowderBlue, Color.Purple, Color.Red, Color.RosyBrown, Color.RoyalBlue, Color.SaddleBrown, Color.Salmon, Color.SeaGreen, Color.Yellow, Color.Khaki, Color.Cyan, Color.DarkMagenta, Color.DarkKhaki, Color.DarkGreen, Color.DarkGray, Color.DarkGoldenrod, Color.DarkCyan, Color.DarkBlue, Color.Ivory, Color.Crimson, Color.Cornsilk, Color.CornflowerBlue, Color.Coral, Color.Chocolate, Color.DarkOliveGreen, Color.Chartreuse, Color.BurlyWood, Color.Brown, Color.BlueViolet, Color.Blue, Color.BlanchedAlmond, Color.Black, Color.Bisque, Color.Beige, Color.Azure, Color.Aquamarine, Color.Aqua, Color.AntiqueWhite, Color.AliceBlue, Color.CadetBlue, Color.DarkOrange, Color.YellowGreen, Color.DarkRed, Color.Indigo, Color.IndianRed, Color.DarkOrchid, Color.Honeydew, Color.GreenYellow, Color.Green, Color.Gray, Color.Goldenrod, Color.Gold, Color.GhostWhite, Color.Gainsboro, Color.Fuchsia, Color.ForestGreen, Color.HotPink, Color.Firebrick, Color.FloralWhite, Color.DodgerBlue, Color.DimGray, Color.DeepSkyBlue, Color.DeepPink, Color.DarkViolet, Color.DarkTurquoise, Color.DarkSlateGray, Color.DarkSlateBlue, Color.DarkSeaGreen, Color.DarkSalmon, };

    public static DotGraph DumpEgraphAsDot(IEGraph eGraph, IReadOnlyList<IMatchResult>? matches, Stream output)
    {
        var printer = new EGraphPrinter(eGraph);
        printer.ConvertEGraphAsDot();
        printer.AttachEGraphMatches(matches);
        return printer.SaveToStream(output);
    }

    public DotGraph AttachEGraphMatches(IReadOnlyList<IMatchResult>? matches)
    {
        // System.Reflection.PropertyInfo[] knowcolors = typeof(Color).GetProperties();
        // var random = new Random(123);
        // int count = 0;
        // foreach (var matchResult in matches)
        // {
        //    var (root, env) = (EMatchResult)matchResult;
        //    var rootCluster = ClusterMaps[eGraph.HashCons[root].Find()];
        //    Color color = _knownColors[random.Next(_knownColors.Length - 1)];
        //    rootCluster.Nodes.Add($"m{rootCluster.Id}_{count}", node =>
        //    {
        //        node.Label = $"m {count}:\n root";
        //        node.Color = color;
        //        node.Shape = DotNodeShape.Circle;
        //        node.Style.FillStyle = DotNodeFillStyle.Radial;
        //    });
        //    foreach (var (pattern, matched_node) in env)
        //    {
        //        EClass matcheClass = eGraph.HashCons[matched_node];
        //        if (OpMaps.ContainsKey(matcheClass)) { continue; }
        //        var childeclassCluster = ClusterMaps[matcheClass.Find()];
        //        if (rootCluster == childeclassCluster) { continue; }
        //        childeclassCluster.Nodes.Add($"m{matcheClass.Id}_{count}", node =>
        //        {
        //            node.Label = $"m {count}";
        //            node.Color = color;
        //            node.Shape = DotNodeShape.Circle;
        //            node.Style.FillStyle = DotNodeFillStyle.Radial;
        //        });
        //    }

        // count++;
        // }
        return _dotGraph;
    }
}
