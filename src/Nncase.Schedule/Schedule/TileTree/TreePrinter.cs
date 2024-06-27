// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using VisitorPatternGenerator;

namespace Nncase.Schedule;

public record TileTreePrinterContext(int ParentOpId, IReadOnlyList<string> Names, int Indent)
{
    public static TileTreePrinterContext Default => new(-1, Array.Empty<string>(), 0);
}

public sealed class TileTreePrinter : ITreeNodeVisitor<TileTreePrinterContext, Unit>
{
    private readonly StreamWriter _writer;

    public TileTreePrinter(StreamWriter writer)
    {
        _writer = writer;
    }

    /// <summary>
    /// get the current tileable node domain dims from parent names.
    /// </summary>
    public static string[] MappingDomainDims(ITileAbleNode value, int parentId, IReadOnlyList<string> pnames)
    {
        var names = value.DomainNames.ToArray();

        if (pnames.Any())
        {
            var relation = value.DomainRelation.Relation;
            for (int i = 0; i < relation.Results.Length; i++)
            {
                if (relation.Results[i] is AffineDim dim)
                {
                    names[i] = pnames[dim.Position];
                }
            }
        }

        return names;
    }

    public void Indent(int indent)
    {
        var s = new string(' ', indent);
        _writer.Write(s);
    }

    public Unit Visit(ScopeNode value, TileTreePrinterContext context)
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

    public Unit Visit(TileNode value, TileTreePrinterContext context)
    {
        var (pid, pnames, indent) = context;
        Indent(indent);
        _writer.WriteLine($"# Tile Op {value.OpId} at level {value.Level}");
        Indent(indent);
        _writer.WriteLine($"# Domain Relation {value.DomainRelation}");
        var names = MappingDomainDims(value, pid, pnames);
        Indent(indent);
        var ivs = string.Join(",", Enumerable.Range(0, names.Length).Select(i => $"i{i}"));
        _writer.WriteLine($"for ({ivs}) in range({string.Join(", ", names)}):");
        indent += 2;

        value.Child?.Accept(this, context with { ParentOpId = value.OpId, Names = names, Indent = indent });
        return default;
    }

    public Unit Visit(OpNode value, TileTreePrinterContext context)
    {
        var (pid, pnames, indent) = context;
        var names = MappingDomainDims(value, pid, pnames);
        Indent(indent);
        _writer.WriteLine($"# Compute Op {value.OpId} at level {value.Level}");
        Indent(indent);
        _writer.WriteLine($"# Domain Relation {value.DomainRelation}");
        Indent(indent);
        var ivs = string.Join(", ", Enumerable.Range(0, names.Length).Select(i => $"d{i}"));
        _writer.WriteLine($"with ({string.Join(", ", names)}) as ({ivs}):");
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
}
