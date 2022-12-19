// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
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


namespace Nncase.IR;

internal sealed class ILDotOption
{
    private readonly DotNode? _dotNode;
    private readonly string? _str;

    public DotNode DotNode => _dotNode!;
    public string Str => _str!;

    public bool IsDotNode => _dotNode is not null;

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
}

internal sealed class ILDotPrintVisitor : ExprVisitor<ILDotOption, string>
{
    private bool _display_callable;
    private DotGraph _dotGraph;
    private List<ValueTuple<string, DotGraph>> _subdotGraphs;
    private int _IdCounter = 0;

    public ILDotPrintVisitor(bool display_callable)
    {
        this._display_callable = display_callable;
        _dotGraph = new(directed: true);
        _subdotGraphs = new();
    }

    private BaseFunction? _entryBaseFunc = null;

    /// <inheritdoc/>
    public override ILDotOption Visit(BaseFunction baseFunction)
    {
        _entryBaseFunc ??= baseFunction;
        return base.Visit(baseFunction);
    }

    /// <inheritdoc/>
    public override ILDotOption Visit(PrimFunctionWrapper expr)
    {
        if (!ExpressionMemo.TryGetValue(expr, out var result))
        {
            var id = _IdCounter++;
            string exprId = "\"" + id.ToString() + "\"";
            result = new(expr.Name);
            ExpressionMemo.Add(expr, result);
        }
        return result;
    }

    /// <inheritdoc/>
    public override ILDotOption Visit(TIR.PrimFunction expr)
    {
        if (!ExpressionMemo.TryGetValue(expr, out var result))
        {
            var id = _IdCounter++;
            string exprId = "\"" + id.ToString() + "\"";
            result = new(expr.Name);
            ExpressionMemo.Add(expr, result);
        }
        return result;
    }

    /// <inheritdoc/>
    public override ILDotOption Visit(Fusion expr)
    {
        _entryBaseFunc ??= expr;
        if (!object.ReferenceEquals(_entryBaseFunc, expr))
        {
            if (_display_callable)
            {
                var visitor = new ILDotPrintVisitor(_display_callable);
                visitor.Visit(expr);
                _subdotGraphs.Add((expr.Name, visitor._dotGraph));
                _subdotGraphs.AddRange(visitor._subdotGraphs);
            }
            return new(expr.Name);
        }
        return base.Visit(expr);
    }

    /// <inheritdoc/>
    public override ILDotOption Visit(Function expr)
    {
        _entryBaseFunc ??= expr;
        if (!object.ReferenceEquals(_entryBaseFunc, expr))
        {
            if (_display_callable)
            {
                var visitor = new ILDotPrintVisitor(_display_callable);
                visitor.Visit(expr);
                _subdotGraphs.Add((expr.Name, visitor._dotGraph));
                _subdotGraphs.AddRange(visitor._subdotGraphs);
            }
            return new(expr.Name);
        }
        return base.Visit(expr);
    }

    /// <inheritdoc/>
    public override ILDotOption VisitLeaf(Fusion expr)
    {
        return new(expr.Name);
    }

    /// <inheritdoc/>
    public override ILDotOption VisitLeaf(Function expr)
    {
        return new(expr.Name);
    }

    public override ILDotOption VisitLeaf(Op expr)
    {
        return new(expr.GetType().Name + $"({expr.DisplayProperty()})");
    }

    public override ILDotOption VisitLeaf(Const expr)
    {
        return new(CompilerServices.Print(expr));
    }

    public override ILDotOption VisitLeaf(None expr)
    {
        var id = _IdCounter++;
        string exprId = "\"" + id.ToString() + "\"";
        var dotNode = new DotNode(exprId) { Label = "None", Shape = DotNodeShape.Rectangle };
        _dotGraph.Nodes.Add(dotNode);
        return new(dotNode);
    }

    public override ILDotOption VisitLeaf(Marker expr)
    {
        var id = _IdCounter++;
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
                row.AddCell("Target", cell => cell.PortName = "P0"); // target.
            else
                row.AddCell(target.Str, cell => cell.PortName = "P0");

            if (attr.IsDotNode)
                row.AddCell("Attr", cell => cell.PortName = "P1"); // attr
            else
                row.AddCell(attr.Str, cell => cell.PortName = "P1");
        });
        table.AddRow(row =>
        {
            row.AddCell(expr.CheckedType is null ? "Null" : CompilerServices.Print(expr.CheckedType), cell => cell.ColumnSpan = 3);
        });

        // 3. make crrent node.
        var dotNode = _dotGraph.Nodes.Add(exprId);
        dotNode.ToPlainHtmlNode(table);

        // 4. connect edge.
        if (target.IsDotNode)
            _dotGraph.Edges.Add(target.DotNode, dotNode, edge =>
            {
                edge.Head.Endpoint.Port = new DotEndpointPort("P0");
            });

        if (attr.IsDotNode)
            _dotGraph.Edges.Add(attr.DotNode, dotNode, edge =>
            {
                edge.Head.Endpoint.Port = new DotEndpointPort("P1");
            });

        return new(dotNode);
    }

    public override ILDotOption VisitLeaf(Var expr)
    {
        var id = _IdCounter++;
        string exprId = "\"" + id.ToString() + "\"";
        var dotNode = new DotNode(exprId) { Label = expr.Name, Shape = DotNodeShape.Rectangle };
        _dotGraph.Nodes.Add(dotNode);
        return new(dotNode);
    }

    public override ILDotOption VisitLeaf(Call expr)
    {
        var id = _IdCounter++;
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
            foreach (var (child, arg_name) in expr.Parameters.Zip(expr.Target switch
            {
                Op op => op.Parameters.Select(info => info.Name),
                Fusion fusion => fusion.Parameters.Select(v => v.Name),
                Function func => func.Parameters.Select(v => v.Name),
                PrimFunctionWrapper wrapper => wrapper.Target.Parameters.Select(b => b.Name),
                _ => throw new ArgumentOutOfRangeException()
            }))
            {
                if (child is Const)
                    continue;
                var portName = $"P{count++}";
                row.AddCell(arg_name, cell => cell.PortName = portName);
                connect_list.Add((child, portName));
            }
        });
        table.AddRow(row =>
        {
            row.AddCell(expr.CheckedType is null ? "Null" : CompilerServices.Print(expr.CheckedType), cell => cell.ColumnSpan = connect_list.Count + 2);
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

        return new(dotNode);
    }

    private static void saveToFileCore(DotGraph dotGraph, string name, string prefix, string dumpDir)
    {
        var nprefix = prefix.Any() ? prefix + "_" : prefix;
        string dump_path = Path.Combine(dumpDir, $"{nprefix}{name}.dot");
        dotGraph.Build();
        dotGraph.SaveToFile(dump_path);
    }

    /// <summary>
    /// Save the dot to File
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="prefix">prefix.</param>
    /// <param name="dumpDir">dump dir.</param>
    public void SaveToFile(string name, string prefix, string dumpDir)
    {
        saveToFileCore(_dotGraph, name, prefix, dumpDir);
        foreach (var (sub_name, subGraph) in _subdotGraphs)
        {
            saveToFileCore(subGraph, name + "_" + sub_name, prefix, dumpDir);
        }
    }
}
