// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GiGraph.Dot.Entities.Clusters;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Html.Table;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Types.Colors;
using GiGraph.Dot.Types.Edges;
using GiGraph.Dot.Types.Fonts;
using GiGraph.Dot.Types.Graphs;
using GiGraph.Dot.Types.Nodes;
using GiGraph.Dot.Types.Records;
using GiGraph.Dot.Types.Styling;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Transform;

namespace Nncase.Transform;

public partial class EGraphPrinter
{
    private readonly Dictionary<EClass, DotCluster> ClusterMaps = new Dictionary<EClass, DotCluster>();

    private readonly Dictionary<EClass, string> OpMaps = new();

    private readonly EGraph eGraph;

    private ulong _IdCounter;

    /// <summary>
    /// the expr map to the dot node and html table.
    /// </summary>
    protected readonly Dictionary<ENode, (DotNode, DotHtmlTable)> NodesMap = new(ReferenceEqualityComparer.Instance);

    private readonly DotDumpVisitor visitor = new DotDumpVisitor();

    /// <summary>
    /// Get the dot graph
    /// </summary>
    public readonly DotGraph dotGraph;

    /// <summary>
    /// ctor for egraph
    /// </summary>
    /// <param name="egraph"></param>
    public EGraphPrinter(EGraph egraph)
    {
        _IdCounter = 0;
        dotGraph = new(directed: true);
        dotGraph.Clusters.AllowEdgeClipping = true;
        eGraph = egraph;
        foreach (var eclass in eGraph.Classes)
        {
            if (eclass.Nodes.Count == 1 && eclass.Nodes[0].Expr is Op op)
            {
                if (!OpMaps.ContainsKey(eclass))
                {
                    var name = visitor.Visit(op);
                    OpMaps.Add(eclass, name);
                }
            }
        }
    }

    /// <summary>
    /// convert the egraph to dot graph
    /// </summary>
    /// <returns></returns>
    public DotGraph ConvertEGraphAsDot()
    {
        foreach (var eClass in eGraph.Classes.Where(x => !OpMaps.ContainsKey(x)))
        {
            // make eClass as cluster
            var eclassCluster = dotGraph.Clusters.Add($"{eClass.Id}", cluster =>
           {
               cluster.Style.BorderStyle = DotBorderStyle.Dotted;
               cluster.Label = $"{eClass.Id}";
               cluster.LabelAlignment.Horizontal = GiGraph.Dot.Types.Alignment.DotHorizontalAlignment.Left;
           });
            ClusterMaps.Add(eClass, eclassCluster);

            eclassCluster.Nodes.Add(new DotNode(eclassCluster.Id + "dummy"), node =>
              {
                  node.Label = "";
                  node.Style.Invisible = true;
                  node.Size.Height = 0;
                  node.Size.Width = 0;
              });

            foreach (var enode in eClass.Nodes)
            {
                if (NodesMap.TryGetValue(enode, out var dotnode))
                    continue;
                var id = _IdCounter++;
                string exprId = "\"" + id.ToString() + "\"";

                var table = new DotHtmlTable
                {
                    BorderWidth = 0,
                    CellBorderWidth = 1,
                    CellSpacing = 0,
                    CellPadding = 4
                };

                // 1. the enode type and children.
                table.AddRow(row =>
                {
                    row.AddCell(visitor.Visit(enode.Expr), font: enode.Expr switch
                    {
                        IR.Const => new DotStyledFont(DotFontStyles.Normal, Color.DarkOrange),
                        IR.Call => new DotStyledFont(DotFontStyles.Normal, Color.DarkBlue),
                        IR.Var => new DotStyledFont(DotFontStyles.Normal, Color.BlueViolet),
                        IR.Fusion => new DotStyledFont(DotFontStyles.Normal, Color.MediumSeaGreen),
                        _ => new DotStyledFont(DotFontStyles.Normal)
                    }); // key wrods type.
                    foreach (var (child, i) in enode.Children.Select((c, i) => (c, i)))
                    {
                        var label = $"{child.Find().Id}";
                        if (OpMaps.ContainsKey(child))
                        {
                            label = $"({label.ToString()}) " + OpMaps[child];
                        }

                        row.AddCell(label, cell => cell.PortName = $"P{i}");
                    }
                });
                // 2. when enode.Expr need show checked type.
                if (enode.Expr is Call or Function or Var)
                {
                    table.AddRow(row =>
                    {
                        row.AddCell(CompilerServices.Print(eClass.CheckedType));
                    });
                }

                // var exprNode = eclassCluster.Nodes.Add(exprId);
                var exprNode = eclassCluster.Nodes.Add(exprId);
                exprNode.ToPlainHtmlNode(table);

                NodesMap.Add(enode, (exprNode, table));

                for (int i = 0; i < enode.Children.Count; i++)
                {
                    if (OpMaps.ContainsKey(enode.Children[i]))
                    {
                        continue;
                    }

                    // var pnode =  from pnode in select
                    dotGraph.Edges.Add($"{enode.Children[i].Find().Id}" + "dummy", exprNode, edge =>
                     {
                         edge.Tail.ClusterId = $"{enode.Children[i].Id}";
                         edge.Head.Endpoint.Port = new DotEndpointPort($"P{i}");
                     });
                }
            }
        }

        return dotGraph;
    }

    /// <summary>
    /// Save the DotGraph into file
    /// </summary>
    /// <param name="file">file path.</param>
    /// <returns>this dot graph.</returns>
    public DotGraph SaveToFile(string file)
    {
        if (!file.EndsWith(".dot"))
        {
            file += ".dot";
        }

        var dirName = Path.GetDirectoryName(file);
        if (dirName is not null && dirName != "")
        {
            Directory.CreateDirectory(dirName);
        }

        dotGraph.Build();
        dotGraph.SaveToFile(file);
        return dotGraph;
    }

    /// <summary>
    /// dump egraph as dot graph.
    /// </summary>
    /// <param name="eGraph">egraph.</param>
    /// <param name="file">path.</param>
    /// <returns>Converted Graph.</returns>
    public static DotGraph DumpEgraphAsDot(EGraph eGraph, string file)
    {
        var printer = new EGraphPrinter(eGraph);
        printer.ConvertEGraphAsDot();
        return printer.SaveToFile(file);
    }

    private class DotDumpVisitor : ExprFunctor<string, string>
    {

        private Dictionary<Const, string> _constNames = new();

        public override string Visit(Call expr)
        {
            return expr.GetType().Name;
        }

        public override string Visit(Const expr)
        {
            if (_constNames.TryGetValue(expr, out var name)) { return name; }
            string valueStr = expr switch
            {
                TensorConst tc => tc.Value.Shape.Size <= 8 ? tc.Value.GetArrayString(false) : string.Empty,
                TupleConst tpc => string.Empty,
                _ => throw new ArgumentOutOfRangeException(),
            };
            valueStr = valueStr != string.Empty ? " : " + valueStr : string.Empty;
            name = $"{CompilerServices.Print(expr.CheckedType!)}{valueStr}";
            _constNames.Add(expr, name);
            return name;
        }

        public override string Visit(BaseFunction expr) => $"{expr.GetType().Name} {expr.Name}";

        public override string Visit(Op expr)
        {
            return expr switch
            {
                Unary op => op.UnaryOp.ToString(),
                Binary op => op.BinaryOp.ToString(),
                Reduce op => "Reduce" + op.ReduceOp.ToString(),
                _ => expr.GetType().Name,
            };
        }

        public override string Visit(Var expr) => expr.GetType().Name + " " + expr.Name;

        public override string Visit(Marker expr) => expr.GetType().Name + " " + expr.Name;

        public override string Visit(IR.Tuple expr) => "Tuple";

        public override string Visit(None expr) => "None";
    }
}
