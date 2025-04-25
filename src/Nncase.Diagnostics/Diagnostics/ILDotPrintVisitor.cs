// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using GiGraph.Dot.Entities.Clusters;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Html.Builder;
using GiGraph.Dot.Entities.Html.Table;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Output.Options;
using GiGraph.Dot.Types.Colors;
using GiGraph.Dot.Types.Edges;
using GiGraph.Dot.Types.Fonts;
using GiGraph.Dot.Types.Graphs;
using GiGraph.Dot.Types.Nodes;
using GiGraph.Dot.Types.Records;
using GiGraph.Dot.Types.Styling;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Buffers;

namespace Nncase.Diagnostics;

internal sealed class ILDotOption
{
    private readonly DotNode? _dotNode;
    private readonly string? _str;

    public ILDotOption(DotNode dotNode)
    {
        _dotNode = dotNode;
        _str = null;
    }

    public ILDotOption(string str)
    {
        _dotNode = null;
        _str = str;
    }

    public DotNode DotNode => _dotNode!;

    public string Str => _str!;

    public bool IsDotNode => _dotNode is not null;
}

internal sealed class DotHtmlUnescapedText : GiGraph.Dot.Entities.Html.DotHtmlEntity, GiGraph.Dot.Entities.Html.IDotHtmlEntity
{
    private readonly string _text;

    public DotHtmlUnescapedText(string text)
    {
        _text = text;
    }

    protected override string ToHtml(DotSyntaxOptions options, DotSyntaxRules syntaxRules) => _text;
}

internal sealed class ILDotPrintVisitor : ExprFunctor<ILDotOption, string>
{
    private readonly DotGraph _dotGraph;
    private readonly List<(string, DotGraph)> _subdotGraphs;
    private readonly Dictionary<Expr, ILDotOption> _exprMemo = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<Var, int> _varColorMemo = new(ReferenceEqualityComparer.Instance);
    private readonly PrinterFlags _flags;
    private int _idCounter;

    private BaseFunction? _entryBaseFunc;

    public ILDotPrintVisitor(PrinterFlags flags)
    {
        _flags = flags;
        _dotGraph = new(directed: true);
        _subdotGraphs = new();
    }

    /// <summary>
    /// Save the dot to File.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="prefix">prefix.</param>
    /// <param name="dumpDir">dump dir.</param>
    public void SaveToFile(string name, string prefix, string dumpDir)
    {
        SaveToFileCore(_dotGraph, name, prefix, dumpDir);
        foreach (var (sub_name, subGraph) in _subdotGraphs)
        {
            SaveToFileCore(subGraph, name + "_" + sub_name, prefix, dumpDir);
        }
    }

    public override string DefaultVisitType(IRType type)
    {
        var feedDict = new Dictionary<Expr, string>();

        foreach (var var in CollectShapeExprs(type).OfType<Var>())
        {
            UpdateVarColor(var);
            var exprName = $"%{var.Name}#{var.GlobalVarIndex}";
            feedDict.Add(var, new DotHtmlBuilder().AppendText(exprName, new DotFont(GetVarColor(var))).Build().ToHtml());
        }

        var stream = Stream.Null;
        using var dumpWriter = new IndentedWriter(stream);
        return new ILPrintVisitor(dumpWriter, PrinterFlags.Minimal, feedDict).VisitType(type).AsHtml();
    }

    /// <inheritdoc/>
    protected override ILDotOption VisitPrimFunctionWrapper(PrimFunctionWrapper expr)
    {
        _entryBaseFunc ??= expr;
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            var id = _idCounter++;
            _ = "\"" + id.ToString() + "\"";
            result = new(expr.Name);
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    /// <inheritdoc/>
    protected override ILDotOption VisitPrimFunction(TIR.PrimFunction expr)
    {
        _entryBaseFunc ??= expr;
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            var id = _idCounter++;
            _ = "\"" + id.ToString() + "\"";
            result = new(expr.Name);
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    // todo: fix this
    protected override ILDotOption VisitIf(If expr) => new("if");

    /// <inheritdoc/>
    protected override ILDotOption VisitFusion(Fusion expr)
    {
        _entryBaseFunc ??= expr;
        if (!ReferenceEquals(_entryBaseFunc, expr))
        {
            if (_flags.HasFlag(PrinterFlags.Detailed))
            {
                var visitor = new ILDotPrintVisitor(_flags);
                visitor.Visit(expr);
                _subdotGraphs.Add((expr.Name, visitor._dotGraph));
                _subdotGraphs.AddRange(visitor._subdotGraphs);
            }
        }
        else
        {
            VisitArray(expr.Parameters);
            Visit(expr.Body);
        }

        return new(expr.Name);
    }

    /// <inheritdoc/>
    protected override ILDotOption VisitFunction(Function expr)
    {
        _entryBaseFunc ??= expr;
        if (!ReferenceEquals(_entryBaseFunc, expr))
        {
            if (_flags.HasFlag(PrinterFlags.Detailed))
            {
                var visitor = new ILDotPrintVisitor(_flags);
                visitor.Visit(expr);
                _subdotGraphs.Add((expr.Name, visitor._dotGraph));
                _subdotGraphs.AddRange(visitor._subdotGraphs);
            }
        }
        else
        {
            VisitArray(expr.Parameters);
            Visit(expr.Body);
        }

        return new(expr.Name);
    }

    protected override ILDotOption VisitBufferOf(BufferOf expr)
    {
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            var id = _idCounter++;
            string exprId = "\"" + id.ToString() + "\"";

            var table = new DotHtmlTable
            {
                BorderWidth = 0,
                CellBorderWidth = 1,
                CellSpacing = 0,
            };

            var connect_list = new List<(Expr, string)>();

            // 1. the connect type.
            table.AddRow(row =>
            {
                row.AddCell("BufferOf"); // key wrods type.
                row.AddCell(Visit(expr.Input).Str); // target.
            });

            // 3. make crrent node.
            var dotNode = _dotGraph.Nodes.Add(exprId);
            dotNode.ToPlainHtmlNode(table);

            // 4. connect edge.
            // _dotGraph.Edges.Add(Visit(expr.Input).DotNode, dotNode);
            result = new(dotNode);
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    protected override ILDotOption VisitGrid(Grid expr)
    {
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            var id = _idCounter++;
            string exprId = "\"" + id.ToString() + "\"";

            var table = new DotHtmlTable
            {
                BorderWidth = 0,
                CellBorderWidth = 1,
                CellSpacing = 0,
            };

            var connect_list = new List<(Expr, string)>();

            // 1. the connect type.
            table.AddRow(row =>
            {
                row.AddCell("Grid"); // key wrods type.
                int count = 0;
                foreach (var child in expr.Buffers)
                {
                    var childnode = Visit(child);
                    var portName = $"P{count++}";
                    row.AddCell(childnode.IsDotNode ? string.Empty : childnode.Str, cell => cell.PortName = portName);
                    if (childnode.IsDotNode)
                    {
                        connect_list.Add((child, portName));
                    }
                }
            });

            // 3. make crrent node.
            var dotNode = _dotGraph.Nodes.Add(exprId);
            dotNode.ToPlainHtmlNode(table);

            // 4. connect edge.
            foreach (var (child, port_name) in connect_list)
            {
                _dotGraph.Edges.Add(Visit(child).DotNode, dotNode, edge =>
                {
                    edge.Head.Endpoint.Port = new DotEndpointPort(port_name);
                });
            }

            result = new(dotNode);
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    protected override ILDotOption VisitTuple(IR.Tuple expr)
    {
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            var id = _idCounter++;
            string exprId = "\"" + id.ToString() + "\"";

            var table = new DotHtmlTable
            {
                BorderWidth = 0,
                CellBorderWidth = 1,
                CellSpacing = 0,
            };

            var connect_list = new List<(Expr, string)>();

            // 1. the connect type.
            table.AddRow(row =>
            {
                row.AddCell("Tuple"); // key wrods type.
                int count = 0;
                foreach (var child in expr.Fields)
                {
                    var childnode = Visit(child);
                    var portName = $"P{count++}";
                    row.AddCell(childnode.IsDotNode ? string.Empty : childnode.Str, cell => cell.PortName = portName);
                    if (childnode.IsDotNode)
                    {
                        connect_list.Add((child, portName));
                    }
                }
            });

            // 3. make crrent node.
            var dotNode = _dotGraph.Nodes.Add(exprId);
            dotNode.ToPlainHtmlNode(table);

            // 4. connect edge.
            foreach (var (child, port_name) in connect_list)
            {
                _dotGraph.Edges.Add(Visit(child).DotNode, dotNode, edge =>
                {
                    edge.Head.Endpoint.Port = new DotEndpointPort(port_name);
                });
            }

            result = new(dotNode);
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    protected override ILDotOption VisitOp(Op expr)
    {
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            result = new(expr.GetType().Name + $"({expr.DisplayProperty()})");
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    protected override ILDotOption VisitConst(Const expr)
    {
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            result = new(CompilerServices.Print(expr));
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    protected override ILDotOption VisitNone(None expr)
    {
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            result = new("None");
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    protected override ILDotOption VisitMarker(Marker expr)
    {
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            var id = _idCounter++;
            string exprId = "\"" + id.ToString() + "\"";

            var table = new DotHtmlTable
            {
                BorderWidth = 0,
                CellBorderWidth = 1,
                CellSpacing = 0,
            };
            var target = Visit(expr.Target);
            var attr = Visit(expr.Attribute);

            // 1. the connect type.
            table.AddRow(row =>
            {
                row.AddCell("Marker"); // key wrods type.
                if (target.IsDotNode)
                {
                    row.AddCell("Target", cell => cell.PortName = "P0"); // target.
                }
                else
                {
                    row.AddCell(target.Str, cell => cell.PortName = "P0");
                }

                if (attr.IsDotNode)
                {
                    row.AddCell("Attr", cell => cell.PortName = "P1"); // attr
                }
                else
                {
                    row.AddCell(attr.Str, cell => cell.PortName = "P1");
                }
            });
            table.AddRow(row =>
            {
                var cell = new DotHtmlTableCell();
                cell.SetContent(new DotHtmlUnescapedText(VisitType(expr.CheckedType).AsHtml()));
                cell.ColumnSpan = 3;
                row.Add(cell);
            });

            // 3. make crrent node.
            var dotNode = _dotGraph.Nodes.Add(exprId);
            dotNode.ToPlainHtmlNode(table);

            // 4. connect edge.
            if (target.IsDotNode)
            {
                _dotGraph.Edges.Add(target.DotNode, dotNode, edge =>
                {
                    edge.Head.Endpoint.Port = new DotEndpointPort("P0");
                });
            }

            if (attr.IsDotNode)
            {
                _dotGraph.Edges.Add(attr.DotNode, dotNode, edge =>
                {
                    edge.Head.Endpoint.Port = new DotEndpointPort("P1");
                });
            }

            result = new(dotNode);
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    protected override ILDotOption VisitVar(Var expr)
    {
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            var id = _idCounter++;
            string exprId = "\"" + id.ToString() + "\"";
            var exprName = $"%{expr.Name}#{expr.GlobalVarIndex}";
            var table = new DotHtmlTable();
            table.AddRow(row => { row.AddCell(exprName); });
            table.AddRow(row =>
            {
                var cell = new DotHtmlTableCell();
                cell.SetContent(new DotHtmlUnescapedText(VisitType(expr.TypeAnnotation).AsHtml()));
                row.Add(cell);
            });
            var dotNode = _dotGraph.Nodes.Add(exprId);
            dotNode.ToPlainHtmlNode(table);
            result = new(dotNode);
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    protected override ILDotOption VisitCall(Call expr)
    {
        if (!_exprMemo.TryGetValue(expr, out var result))
        {
            var id = _idCounter++;
            string exprId = "\"" + id.ToString() + "\"";

            var table = new DotHtmlTable
            {
                BorderWidth = 0,
                CellBorderWidth = 1,
                CellSpacing = 0,
            };

            var connect_list = new List<(Expr, string)>();

            // 1. the connect type.
            table.AddRow(row =>
            {
                row.AddCell("Call"); // key wrods type.
                row.AddCell(Visit(expr.Target).Str); // target.
                int count = 0;
                foreach (var (child, arg_name) in expr.Arguments.ToArray().Zip(expr.Target switch
                {
                    Op op => op.Parameters.Select(info => info.Name),
                    Fusion fusion => fusion.Parameters.AsValueEnumerable().Select(v => v.Name).ToArray(),
                    Function func => func.Parameters.AsValueEnumerable().Select(v => v.Name).ToArray(),
                    PrimFunctionWrapper wrapper => wrapper.Target.Parameters.AsValueEnumerable().Select(x => x.Name).ToArray(),
                    _ => throw new NotSupportedException($"Target type {expr.Target.GetType()} is not supported."),
                }))
                {
                    if (child is None)
                    {
                        continue;
                    }

                    var portName = $"P{count++}";
                    row.AddCell(child switch { Const c => c.CheckedType.ToString(), _ => arg_name }, cell => cell.PortName = portName);
                    connect_list.Add((child, portName));
                }
            });
            table.AddRow(row =>
            {
                var cell = new DotHtmlTableCell();
                cell.SetContent(new DotHtmlUnescapedText(VisitType(expr.CheckedType).AsHtml()));
                cell.ColumnSpan = connect_list.Count + 2;
                row.Add(cell);
            });

            // 3. make crrent node.
            var dotNode = _dotGraph.Nodes.Add(exprId);
            dotNode.ToPlainHtmlNode(table);

            // 4. connect edge.
            foreach (var (child, port_name) in connect_list)
            {
                if (child is BaseFunction or Const or If)
                {
                    continue;
                }

                _dotGraph.Edges.Add(Visit(child).DotNode, dotNode, edge =>
                {
                    edge.Head.Endpoint.Port = new DotEndpointPort(port_name);
                });
            }

            result = new(dotNode);
            _exprMemo.Add(expr, result);
        }

        return result;
    }

    private static void SaveToFileCore(DotGraph dotGraph, string name, string prefix, string dumpDir)
    {
        var nprefix = prefix.Any() ? prefix + "_" : prefix;
        string dump_path = Path.Combine(dumpDir, $"{nprefix}{name}.dot");
        dotGraph.Build();
        dotGraph.SaveToFile(dump_path);
    }

    private void VisitArray<T>(ReadOnlySpan<T> exprs)
        where T : Expr
    {
        foreach (var expr in exprs)
        {
            Visit(expr);
        }
    }

    private void UpdateVarColor(Var expr)
    {
        if (!_varColorMemo.TryGetValue(expr, out var _))
        {
            _varColorMemo.Add(expr, _varColorMemo.Keys.Count);
        }
    }

    private DotColor GetVarColor(Var expr)
    {
        return new DotColor(Utility.BaseColors[_varColorMemo[expr] % Utility.BaseColors.Length]);
    }

    private List<Expr> CollectShapeExprs(IRType irType)
    {
        var shapes = new List<Shape>();

        void InnerCollect(IRType type)
        {
            switch (type)
            {
                case TensorType t:
                    shapes.Add(t.Shape);
                    break;
                case DistributedType dt:
                    shapes.Add(dt.TensorType.Shape);
                    break;
                case TupleType tt:
                    foreach (var field in tt.Fields)
                    {
                        InnerCollect(field);
                    }

                    break;
            }
        }

        InnerCollect(irType);
        var collector = new ShapeExprWalker();
        foreach (var shape in shapes)
        {
            collector.Visit(shape);
        }

        return collector.ExprMemo.Keys.ToList();
    }
}

internal sealed class ShapeExprWalker : ExprWalker
{
    public ShapeExprWalker()
        : base(false, false)
    {
    }
}
