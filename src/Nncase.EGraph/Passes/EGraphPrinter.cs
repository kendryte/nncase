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
using Nncase.Passes;

namespace Nncase.Passes;

public partial class EGraphPrinter
{
    /// <summary>
    /// Get the dot graph.
    /// </summary>
    private readonly DotGraph _dotGraph;

    private readonly Dictionary<EClass, DotCluster> _clusterMaps = new Dictionary<EClass, DotCluster>();

    private readonly Dictionary<EClass, string> _opMaps = new();

    private readonly IEGraph _eGraph;

    private readonly DotDumpVisitor _visitor = new DotDumpVisitor();

    private ulong _idCounter;

    /// <summary>
    /// Initializes a new instance of the <see cref="EGraphPrinter"/> class.
    /// ctor for egraph.
    /// </summary>
    public EGraphPrinter(IEGraph egraph)
    {
        _idCounter = 0;
        _dotGraph = new(directed: true);
        _dotGraph.Clusters.AllowEdgeClipping = true;
        _eGraph = egraph;
        foreach (var eclass in _eGraph.Classes)
        {
            if (eclass.Nodes.Count == 1 && eclass.Nodes[0].Expr is Op op)
            {
                if (!_opMaps.ContainsKey(eclass))
                {
                    var name = _visitor.Visit(op);
                    _opMaps.Add(eclass, name);
                }
            }
        }
    }

    /// <summary>
    /// Gets the expr map to the dot node and html table.
    /// </summary>
    protected Dictionary<ENode, (DotNode Node, DotHtmlTable Table)> NodesMap { get; } = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    /// dump egraph as dot graph.
    /// </summary>
    /// <param name="eGraph">egraph.</param>
    /// <param name="output">Output stream.</param>
    /// <returns>Converted Graph.</returns>
    public static DotGraph DumpEgraphAsDot(IEGraph eGraph, Stream output)
    {
        var printer = new EGraphPrinter(eGraph);
        printer.ConvertEGraphAsDot();
        return printer.SaveToStream(output);
    }

    /// <summary>
    /// dump egraph as dot graph.
    /// </summary>
    /// <param name="eGraph">egraph.</param>
    /// <param name="file">path.</param>
    /// <returns>Converted Graph.</returns>
    public static DotGraph DumpEgraphAsDot(IEGraph eGraph, string file)
    {
        var printer = new EGraphPrinter(eGraph);
        printer.ConvertEGraphAsDot();
        return printer.SaveToFile(file);
    }

    /// <summary>
    /// convert the egraph to dot graph.
    /// </summary>
    public DotGraph ConvertEGraphAsDot()
    {
        foreach (var eClass in _eGraph.Classes.Where(x => !_opMaps.ContainsKey(x)))
        {
            // make eClass as cluster
            var eclassCluster = _dotGraph.Clusters.Add($"{eClass.Id}", cluster =>
           {
               cluster.Style.BorderStyle = DotBorderStyle.Dotted;
               cluster.Label = $"{eClass.Id}";
               cluster.LabelAlignment.Horizontal = GiGraph.Dot.Types.Alignment.DotHorizontalAlignment.Left;
           });
            _clusterMaps.Add(eClass, eclassCluster);

            eclassCluster.Nodes.Add(new DotNode(eclassCluster.Id + "dummy"), node =>
              {
                  node.Label = string.Empty;
                  node.Style.Invisible = true;
                  node.Size.Height = 0;
                  node.Size.Width = 0;
              });

            foreach (var enode in eClass.Nodes)
            {
                if (NodesMap.TryGetValue(enode, out var dotnode))
                {
                    continue;
                }

                var id = _idCounter++;
                string exprId = "\"" + id.ToString() + "\"";

                var table = new DotHtmlTable
                {
                    BorderWidth = 0,
                    CellBorderWidth = 1,
                    CellSpacing = 0,
                    CellPadding = 4,
                };

                // 1. the enode type and children.
                table.AddRow(row =>
                {
                    var font = enode.Expr switch
                    {
                        IR.Const => new DotStyledFont(DotFontStyles.Normal, Color.DarkOrange),
                        IR.Call => new DotStyledFont(DotFontStyles.Normal, Color.DarkBlue),
                        IR.Var => new DotStyledFont(DotFontStyles.Normal, Color.BlueViolet),
                        IR.Fusion => new DotStyledFont(DotFontStyles.Normal, Color.MediumSeaGreen),
                        _ => new DotStyledFont(DotFontStyles.Normal),
                    };
                    row.AddCell(_visitor.Visit(enode.Expr), font); // key wrods type.
                    foreach (var (child, i) in enode.Children.Select((c, i) => (c, i)))
                    {
                        var label = $"{child.Find().Id}";
                        if (_opMaps.ContainsKey(child))
                        {
                            label = $"({label.ToString()}) " + _opMaps[child];
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
                    if (_opMaps.ContainsKey(enode.Children[i]))
                    {
                        continue;
                    }

                    // var pnode =  from pnode in select
                    _dotGraph.Edges.Add($"{enode.Children[i].Find().Id}" + "dummy", exprNode, edge =>
                     {
                         edge.Tail.ClusterId = $"{enode.Children[i].Id}";
                         edge.Head.Endpoint.Port = new DotEndpointPort($"P{i}");
                     });
                }
            }
        }

        return _dotGraph;
    }

    /// <summary>
    /// Save the _dotGraph into stream.
    /// </summary>
    /// <param name="output">Output stream.</param>
    /// <returns>this dot graph.</returns>
    public DotGraph SaveToStream(Stream output)
    {
        using (var writer = new StreamWriter(output, leaveOpen: true))
        {
            _dotGraph.Build(writer);
        }

        return _dotGraph;
    }

    /// <summary>
    /// Save the _dotGraph into file.
    /// </summary>
    /// <param name="file">Output file.</param>
    /// <returns>this dot graph.</returns>
    public DotGraph SaveToFile(string file)
    {
        if (!file.EndsWith(".dot"))
        {
            file += ".dot";
        }

        var dirName = Path.GetDirectoryName(file);
        if (dirName is not null && dirName != string.Empty)
        {
            Directory.CreateDirectory(dirName);
        }

        _dotGraph.Build();
        _dotGraph.SaveToFile(file);
        return _dotGraph;
    }

    private class DotDumpVisitor : ExprFunctor<string, string>
    {
        private readonly Dictionary<Const, string> _constNames = new();

        protected override string VisitCall(Call expr)
        {
            return expr.GetType().Name;
        }

        protected override string VisitConst(Const expr)
        {
            if (_constNames.TryGetValue(expr, out var name))
            {
                return name;
            }

            string valueStr = expr switch
            {
                TensorConst tc => tc.Value.Shape.Size <= 8 ? tc.Value.GetArrayString(false) : string.Empty,
                TupleConst => string.Empty,
                _ => throw new ArgumentOutOfRangeException(nameof(expr)),
            };
            valueStr = valueStr != string.Empty ? " : " + valueStr : string.Empty;
            name = $"{CompilerServices.Print(expr.CheckedType!)}{valueStr}";
            _constNames.Add(expr, name);
            return name;
        }

        protected override string VisitBaseFunction(BaseFunction expr) => $"{expr.GetType().Name} {expr.Name}";

        protected override string VisitOp(Op expr)
        {
            return $"{expr.GetType().Name}({expr.DisplayProperty()})";
        }

        protected override string VisitVar(Var expr) => expr.GetType().Name + " " + expr.Name;

        protected override string VisitMarker(Marker expr) => expr.GetType().Name + " " + expr.Name;

        protected override string VisitIf(IR.If expr) => "If";

        protected override string VisitTuple(IR.Tuple expr) => "Tuple";

        protected override string VisitNone(None expr) => "None";
    }
}
