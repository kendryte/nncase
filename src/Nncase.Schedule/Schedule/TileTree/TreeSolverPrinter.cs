// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.CodeDom.Compiler;
using System.Reactive;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileTree;

public sealed class TreeSolverPrinter : TreeSolverBase, ITreeNodeVisitor<Unit, Unit>
{
    public TreeSolverPrinter(StreamWriter writer, Assignment? solution, Solver solver, IntExpr one, IntExpr zero, IntExpr elem, Dictionary<OpNode, OpNodeInfo> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo> levelBufferInfos, Dictionary<ITileAbleNode, DomainInfo> domainInfos, ITargetOptions targetOptions)
        : base(solver, primitiveBufferInfo, levelBufferInfos, domainInfos, targetOptions)
    {
        Writer = new IndentedTextWriter(writer, "  ");
        Solution = solution;
    }

    public IndentedTextWriter Writer { get; }

    public Assignment? Solution { get; }

    public static void WriteIntExprVector(IndentedTextWriter writer, string prefix, PropagationBaseObject[] intExprs, Assignment? solution = null)
    {
        writer.WriteLine($"{prefix}: ");
        writer.Indent++;
        for (int i = 0; i < intExprs.Length; i++)
        {
            string value = string.Empty;
            if (solution is Assignment assignment && intExprs[i] is IntExpr expr)
            {
                value = $"= {assignment.Value(expr.Var())}";
            }

            writer.WriteLine($"{i}: {intExprs[i].ToSimplifyString()} {value}");
        }

        writer.Indent--;
    }

    public static void WriteIntExprMatrix(IndentedTextWriter writer, string prefix, PropagationBaseObject[][] intMatrix, Assignment? solution = null)
    {
        writer.WriteLine($"{prefix}:");
        writer.Indent++;
        for (int i = 0; i < intMatrix.Length; i++)
        {
            var vector = intMatrix[i];
            WriteIntExprVector(writer, i.ToString(), vector, solution);
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
        WriteIntExprVector(Writer, "domainInfo", TileableNodeMemo[value].TileVars, Solution);

        WriteIntExprMatrix(Writer, "BackWardExtents", TileNodeMemo[value].BackWardExtents, Solution);

        Writer.WriteLine($"BufferInfo:");
        Writer.Indent++;
        foreach (var (bid, info) in TileNodeMemo[value].BufferInfoMap)
        {
            Writer.WriteLine($"{bid}:");
            Writer.Indent++;
            WriteIntExprMatrix(Writer, "Shapes", info.Shapes, Solution);
            WriteIntExprVector(Writer, "SizeVars", info.SizeVars, Solution);
            WriteIntExprVector(Writer, "SizeExprs", info.SizeExprs, Solution);
            WriteIntExprVector(Writer, "Writes", info.Writes, Solution);
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
        WriteIntExprMatrix(Writer, "Shapes", OpNodeMemo[value].Shapes, Solution);
        Writer.Indent--;
        Writer.WriteLine($"\"\"\"");

        var ivs = string.Join(", ", Enumerable.Range(0, value.DimNames.Length).Select(i => $"d{i}"));
        Writer.WriteLine($"with ({string.Join(", ", value.DimNames)}) as ({ivs}):");
        Writer.Indent++;
        var set = value.Dependences.ToDictionary(d => d.Index, d => d.Node);
        for (int i = 0; i < value.ReadAccesses.Length; i++)
        {
            Writer.Write($"read_{i} = ");
            Writer.Write(value.ReadAccesses[i]);
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
        Writer.WriteLine(value.WriteAccess);
        Writer.Indent--;

        return default;
    }
}
