using System;
using System.Collections.Generic;
using System.Drawing;
using Nncase.Transform;
using Nncase.Transform.Pattern;
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

namespace Nncase.Transform
{
    public partial class EGraphPrinter
    {
        private readonly Color[] KnownColors = new Color[] { Color.MediumAquamarine, Color.MediumBlue, Color.MediumOrchid, Color.MediumPurple, Color.MediumSeaGreen, Color.MediumSlateBlue, Color.MediumSpringGreen, Color.Maroon, Color.MediumTurquoise, Color.MidnightBlue, Color.MintCream, Color.MistyRose, Color.Moccasin, Color.NavajoWhite, Color.Navy, Color.OldLace, Color.MediumVioletRed, Color.Magenta, Color.Linen, Color.LimeGreen, Color.LavenderBlush, Color.LawnGreen, Color.LemonChiffon, Color.LightBlue, Color.LightCoral, Color.LightCyan, Color.LightGoldenrodYellow, Color.LightGray, Color.LightGreen, Color.LightPink, Color.LightSalmon, Color.LightSeaGreen, Color.LightSkyBlue, Color.LightSlateGray, Color.LightSteelBlue, Color.LightYellow, Color.Lime, Color.Olive, Color.OliveDrab, Color.Orange, Color.OrangeRed, Color.Silver, Color.SkyBlue, Color.SlateBlue, Color.SlateGray, Color.Snow, Color.SpringGreen, Color.SteelBlue, Color.Tan, Color.Teal, Color.Thistle, Color.Tomato, Color.Transparent, Color.Turquoise, Color.Violet, Color.Wheat, Color.White, Color.WhiteSmoke, Color.Sienna, Color.Lavender, Color.SeaShell, Color.SandyBrown, Color.Orchid, Color.PaleGoldenrod, Color.PaleGreen, Color.PaleTurquoise, Color.PaleVioletRed, Color.PapayaWhip, Color.PeachPuff, Color.Peru, Color.Pink, Color.Plum, Color.PowderBlue, Color.Purple, Color.Red, Color.RosyBrown, Color.RoyalBlue, Color.SaddleBrown, Color.Salmon, Color.SeaGreen, Color.Yellow, Color.Khaki, Color.Cyan, Color.DarkMagenta, Color.DarkKhaki, Color.DarkGreen, Color.DarkGray, Color.DarkGoldenrod, Color.DarkCyan, Color.DarkBlue, Color.Ivory, Color.Crimson, Color.Cornsilk, Color.CornflowerBlue, Color.Coral, Color.Chocolate, Color.DarkOliveGreen, Color.Chartreuse, Color.BurlyWood, Color.Brown, Color.BlueViolet, Color.Blue, Color.BlanchedAlmond, Color.Black, Color.Bisque, Color.Beige, Color.Azure, Color.Aquamarine, Color.Aqua, Color.AntiqueWhite, Color.AliceBlue, Color.CadetBlue, Color.DarkOrange, Color.YellowGreen, Color.DarkRed, Color.Indigo, Color.IndianRed, Color.DarkOrchid, Color.Honeydew, Color.GreenYellow, Color.Green, Color.Gray, Color.Goldenrod, Color.Gold, Color.GhostWhite, Color.Gainsboro, Color.Fuchsia, Color.ForestGreen, Color.HotPink, Color.Firebrick, Color.FloralWhite, Color.DodgerBlue, Color.DimGray, Color.DeepSkyBlue, Color.DeepPink, Color.DarkViolet, Color.DarkTurquoise, Color.DarkSlateGray, Color.DarkSlateBlue, Color.DarkSeaGreen, Color.DarkSalmon, };

        public DotGraph ConvertEGraphAsDot(EGraph eGraph, List<EMatchResult> matches)
        {
            DotGraph g = ConvertEGraphAsDot(eGraph);
            System.Reflection.PropertyInfo[] knowcolors = typeof(Color).GetProperties();
            var random = new Random(123);
            int count = 0;
            foreach (var (root, env) in matches)
            {
                var eclassCluster = _classes[eGraph.Nodes[root]];
                Color color = KnownColors[random.Next(KnownColors.Length - 1)];
                eclassCluster.Nodes.Add($"m{eclassCluster.Id}_{count}", node =>
                {
                    node.Label = $"m {count}:\n root";
                    node.Color = color;
                    node.Shape = DotNodeShape.Circle;
                });
                foreach (var (wc, wcNode) in env)
                {
                    EClass matcheClass = eGraph.Nodes[wcNode];
                    var childeclassCluster = _classes[matcheClass];
                    childeclassCluster.Nodes.Add($"m{matcheClass.Id}_{count}", node =>
                    {
                        node.Label = $"m {count} :\n {wc.Name}";
                        node.Color = color;
                        node.Shape = DotNodeShape.Circle;
                    });
                }
            }
            return g;
        }

        public static DotGraph DumpEgraphAsDot(EGraph eGraph, List<EMatchResult> matches, string file)
        {
            var printer = new EGraphPrinter();
            var g = printer.ConvertEGraphAsDot(eGraph, matches);
            return printer.SaveToFile(g, file);
        }

    }

}