// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using System.Text;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using VisitorPatternGenerator;

namespace Nncase.Schedule.TileTree;

public record TreePrinterContext(int ParentOpId, IReadOnlyList<string> Names)
{
    public static TreePrinterContext Default => new(-1, Array.Empty<string>());
}

public sealed class TreePrinter : ITreeNodeVisitor<TreePrinterContext, Unit>
{
    public TreePrinter(StreamWriter writer)
    {
        Writer = new(writer, "  ");
    }

    public System.CodeDom.Compiler.IndentedTextWriter Writer { get; }

    /// <summary>
    /// get the current tileable node domain dims from parent names.
    /// </summary>
    public static string[] MappingDomainDims(ITileAbleNode value, int parentId, IReadOnlyList<string> pnames)
    {
        var names = value.DimNames.ToArray();

        if (pnames.Any())
        {
            var relation = value.DomainRelation.Map;
            for (int i = 0; i < relation.Results.Length; i++)
            {
                if (relation.Results[i] is { Offset: AffineDim dim, Extent: AffineExtent ext } && dim.Position == ext.Position)
                {
                    names[i] = pnames[dim.Position];
                }
            }
        }

        return names;
    }

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

    public Unit Visit(ScopeNode value, TreePrinterContext context)
    {
        Writer.WriteLine($"# scope");
        foreach (var child in value.Children)
        {
            child.Accept(this, context);
        }

        return default;
    }

    public Unit Visit(TileNode value, TreePrinterContext context)
    {
        var (pid, pnames) = context;
        Writer.WriteLine($"# Tile Op {value.OpId} at level {value.Level}");
        Writer.WriteLine($"# Domain Relation {value.DomainRelation}");
        var names = MappingDomainDims(value, pid, pnames);
        var ivs = string.Join(",", Enumerable.Range(0, names.Length).Select(i => $"i{i}"));
        Writer.WriteLine($"for ({ivs}) in range({string.Join(", ", names)}):");
        Writer.Indent++;
        value.Child?.Accept(this, context with { ParentOpId = value.OpId, Names = names });
        Writer.Indent--;
        return default;
    }

    public Unit Visit(OpNode value, TreePrinterContext context)
    {
        var (pid, pnames) = context;
        var names = MappingDomainDims(value, pid, pnames);
        Writer.WriteLine($"# Compute Op {value.OpId} at level {value.Level}");
        Writer.WriteLine($"# Domain Relation {value.DomainRelation}");
        Writer.WriteLine($"# Domain Bounds [{string.Join(", ", value.DomainBounds)}]");
        var ivs = string.Join(", ", Enumerable.Range(0, names.Length).Select(i => $"d{i}"));
        Writer.WriteLine($"with ({string.Join(", ", names)}) as ({ivs}):");
        Writer.Indent++;

        var set = value.Dependences.ToDictionary(d => d.Index, d => d.Node);
        for (int i = 0; i < value.Reads.Length; i++)
        {
            Writer.Write($"read_{i} = ");
            Writer.Write(value.Reads[i]);
            if (set.ContainsKey(i))
            {
                Writer.WriteLine($" @ Op{set[i].OpId}");
            }
            else
            {
                Writer.WriteLine();
            }
        }

        Writer.Write("write = ");
        Writer.WriteLine(value.Write);
        Writer.Indent--;
        return default;
    }
}
