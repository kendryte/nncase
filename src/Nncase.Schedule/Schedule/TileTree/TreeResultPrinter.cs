// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using System.Text;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using VisitorPatternGenerator;

namespace Nncase.Schedule;

public sealed class TileTreeResultPrinter : ITreeNodeVisitor<TileTreeResultPrinter.Context, Unit>
{
    private readonly StreamWriter _writer;
    private readonly string[] _ivNames = new string[] { "i", "j", "k", "l", "m", "n", "o" };

    public TileTreeResultPrinter(StreamWriter writer, Dictionary<ITileAbleNode, DomainDimAssignment> tileableNodeMemo, Dictionary<TileNode, TileNodeAssignment> tileNodeMemo)
    {
        _writer = writer;
        TileableNodeMemo = tileableNodeMemo;
        TileNodeMemo = tileNodeMemo;
    }

    public Dictionary<ITileAbleNode, DomainDimAssignment> TileableNodeMemo { get; }

    public Dictionary<TileNode, TileNodeAssignment> TileNodeMemo { get; }

    public void Indent(int indent)
    {
        var s = new string(' ', indent);
        _writer.Write(s);
    }

    public Unit Visit(ScopeNode value, Context context)
    {
        var indent = context.Indent;
        Indent(indent);
        _writer.WriteLine($"# scope");
        foreach (var child in value.Children)
        {
            child.Accept(this, context with { Indent = indent });
        }

        return default;
    }

    public Unit Visit(TileNode value, Context context)
    {
        var indent = context.Indent;
        Indent(indent);
        _writer.WriteLine($"# Tile Op {value.OpId} at level {value.Level}");
        Indent(indent);
        _writer.WriteLine($"# Domain Relation {value.DomainRelation}");
        var names = TileableNodeMemo[value].DimNames;
        var ivs = Enumerable.Range(0, names.Length).Select(i => $"{_ivNames[i]}{value.Level}").ToArray();
        for (int i = 0; i < names.Length; i++)
        {
            Indent(indent);
            _writer.WriteLine($"for {ivs[i]} in range(0, {names[i]}, {TileableNodeMemo[value].TileVars[i]}):");
            indent += 2;
        }

        value.Child?.Accept(this, context with { Indent = indent });
        return default;
    }

    public Unit Visit(OpNode value, Context context)
    {
        var indent = context.Indent;
        var names = TileableNodeMemo[value].DimNames;
        Indent(indent);
        _writer.WriteLine($"# Compute Op {value.OpId} at level {value.Level}");
        Indent(indent);
        _writer.WriteLine($"# Domain Relation {value.DomainRelation}");
        Indent(indent);
        var ivs = string.Join(",", Enumerable.Range(0, names.Length).Select(i => $"{_ivNames[i]}{value.Level}"));
        _writer.WriteLine($"for ({ivs}) in [({string.Join(", ", TileableNodeMemo[value].TileVars)})]:");
        indent += 2;

        var set = value.Dependences.ToDictionary(d => d.Index, d => d.Node);
        for (int i = 0; i < value.Reads.Length; i++)
        {
            Indent(indent);

            _writer.Write($"read_{i} = ");
            _writer.Write(value.Reads[i]);
            if (set.ContainsKey(i))
            {
                _writer.WriteLine($" @ Op{set[i].OpId}");
            }
            else
            {
                _writer.WriteLine();
            }
        }

        Indent(indent);
        _writer.Write("write = ");
        _writer.WriteLine(value.Write);
        return default;
    }

    public sealed record Context(int Indent, IReadOnlyList<int> DomainBounds)
    {
        public static readonly Context Default = new Context(0, Array.Empty<int>());
    }
}
