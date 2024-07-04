// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.CodeDom.Compiler;
using System.Reactive;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileTree;

public sealed class TreeSolverPrinter : TreeSolverBase, ITreeNodeVisitor<Unit, Unit>
{
    public TreeSolverPrinter(StreamWriter writer, Solver solver, IntExpr one, IntExpr zero, IntExpr elem, Dictionary<OpNode, OpNodeInfo> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo> levelBufferInfos, Dictionary<ITileAbleNode, DomainInfo> domainInfos, ITargetOptions targetOptions)
        : base(solver, one, zero, elem, primitiveBufferInfo, levelBufferInfos, domainInfos, targetOptions)
    {
        Writer = new IndentedTextWriter(writer, "  ");
    }

    public IndentedTextWriter Writer { get; }

    public static void WriteIntExprVector(IndentedTextWriter writer, string prefix, PropagationBaseObject[] intExprs)
    {
        writer.WriteLine($"{prefix}: ");
        writer.Indent++;
        for (int i = 0; i < intExprs.Length; i++)
        {
            writer.WriteLine($"{i}: {intExprs[i].ToSimplifyString()}");
        }

        writer.Indent--;
    }

    public static void WriteIntExprMatrix(IndentedTextWriter writer, string prefix, PropagationBaseObject[][] intMatrix)
    {
        writer.WriteLine($"{prefix}:");
        writer.Indent++;
        for (int i = 0; i < intMatrix.Length; i++)
        {
            var vector = intMatrix[i];
            writer.WriteLine($"{i}: [{string.Join(", ", vector.Select(i => i.ToSimplifyString()))}]");
        }

        writer.Indent--;
    }

    public Unit Visit(ScopeNode value, Unit context)
    {
        Writer.WriteLine($"# scope");
        foreach (var child in value.Children)
        {
            child.Accept(this, context);
        }

        return default;
    }

    public Unit Visit(TileNode value, Unit context)
    {
        Writer.WriteLine($"\"\"\"");
        Writer.Indent++;
        Writer.WriteLine($"Tile Op {value.OpId} at level {value.Level}");
        Writer.WriteLine($"Domain Relation {value.DomainRelation}");
        WriteDomainInfo(TileableNodeMemo[value]);

        WriteIntExprMatrix(Writer, "DomainExtents", TileNodeMemo[value].DomainExtents);

        Writer.WriteLine($"BufferInfo:");
        Writer.Indent++;
        foreach (var (bid, info) in TileNodeMemo[value].BufferInfoMap)
        {
            Writer.WriteLine($"{bid}:");
            Writer.Indent++;
            WriteIntExprMatrix(Writer, "Shapes", info.Shapes);
            WriteIntExprVector(Writer, "Sizes", info.Sizes);
            WriteIntExprVector(Writer, "Writes", info.Writes);
            Writer.Indent--;
        }

        Writer.Indent--;
        Writer.Indent--;
        Writer.WriteLine($"\"\"\"");

        var ivs = string.Join(",", Enumerable.Range(0, value.DimNames.Length).Select(i => $"i{i}"));
        Writer.WriteLine($"for ({ivs}) in range({string.Join(", ", value.DimNames)}):");
        Writer.Indent++;
        value.Child.Accept(this, context);
        Writer.Indent--;

        return default;
    }

    public Unit Visit(OpNode value, Unit context)
    {
        Writer.WriteLine($"\"\"\"");
        Writer.Indent++;

        Writer.WriteLine($"Compute Op {value.OpId} at level {value.Level}");
        Writer.WriteLine($"Domain Relation {value.DomainRelation}");
        WriteIntExprMatrix(Writer, "Shapes", OpNodeMemo[value].Shapes);
        Writer.Indent--;
        Writer.WriteLine($"\"\"\"");

        var ivs = string.Join(", ", Enumerable.Range(0, value.DimNames.Length).Select(i => $"d{i}"));
        Writer.WriteLine($"with ({string.Join(", ", value.DimNames)}) as ({ivs}):");
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

    private void WriteDomainInfo(DomainInfo domainInfo)
    {
        Writer.WriteLine($"domainInfo: ");
        Writer.Indent++;
        for (int i = 0; i < domainInfo.TileVars.Length; i++)
        {
            Writer.WriteLine($"{i}, {domainInfo.TileVars[i].ToSimplifyString()}");
        }

        Writer.Indent--;
    }
}
