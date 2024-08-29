// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.CodeDom.Compiler;
using System.Reactive;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileTree;

public sealed class TreeSolverPrinter : TreeSolverBase, ITreeNodeVisitor<IndentedTextWriter, Unit>
{
    public TreeSolverPrinter(Assignment? solution, Solver solver, Dictionary<OpNode, OpNodeInfo> opNodeMemo, Dictionary<TileNode, TileNodeInfo> tileNodeMemo, Dictionary<ITileAbleNode, DomainInfo> tileableNodeMemo, ICpuTargetOptions targetOptions)
        : base(solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions)
    {
        Solution = solution;
    }

    public Assignment? Solution { get; }

    public static void WriteIntExpr(IndentedTextWriter writer, string prefix, PropagationBaseObject intExpr, Assignment? solution = null)
    {
        string value = string.Empty;
        if (solution is Assignment assignment && intExpr is IntExpr expr)
        {
            value = $"= {assignment.Value(expr.Var())}";
        }

        writer.WriteLine($"{prefix}: {intExpr.ToSimplifyString()} {value}");
    }

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

    public Unit Visit(ScopeNode value, IndentedTextWriter writer)
    {
        writer.WriteLine($"# scope");
        foreach (var child in value.Children)
        {
            child.Accept(this, writer);
        }

        return default;
    }

    public Unit Visit(TileNode value, IndentedTextWriter writer)
    {
        writer.WriteLine($"\"\"\"");
        writer.Indent++;
        writer.WriteLine($"Tile Op {value.OpId} at level {value.Level}");
        writer.WriteLine($"Domain Relation {value.DomainRelation}");
        WriteIntExprVector(writer, "domainInfo", TileableNodeMemo[value].TileVars, Solution);

        WriteIntExprMatrix(writer, "BackWardExtents", TileNodeMemo[value].BackWardExtents, Solution);

        writer.WriteLine($"BufferInfo:");
        writer.Indent++;
        foreach (var (bid, info) in TileNodeMemo[value].BufferInfoMap)
        {
            writer.WriteLine($"{bid}:");
            writer.Indent++;
            WriteIntExprMatrix(writer, "Shapes", info.Shapes, Solution);
            WriteIntExprVector(writer, "SizeVars", info.SizeVars, Solution);
            WriteIntExprVector(writer, "SizeExprs", info.SizeExprs, Solution);
            WriteIntExprVector(writer, "Writes", info.Writes, Solution);
            writer.Indent--;
        }

        writer.Indent--;
        writer.Indent--;
        writer.WriteLine($"\"\"\"");

        var ivs = string.Join(",", Enumerable.Range(0, value.DimNames.Length).Select(i => $"i{i}"));
        writer.WriteLine($"for ({ivs}) in range({string.Join(", ", value.DimNames)}):");
        writer.Indent++;
        value.Child.Accept(this, writer);
        writer.Indent--;

        return default;
    }

    public Unit Visit(OpNode value, IndentedTextWriter writer)
    {
        writer.WriteLine($"\"\"\"");
        writer.Indent++;

        writer.WriteLine($"Compute Op {value.OpId} at level {value.Level}");
        writer.WriteLine($"Domain Relation {value.DomainRelation}");
        WriteIntExprMatrix(writer, "Shapes", OpNodeMemo[value].Shapes, Solution);
        writer.Indent--;
        writer.WriteLine($"\"\"\"");

        var ivs = string.Join(", ", Enumerable.Range(0, value.DimNames.Length).Select(i => $"d{i}"));
        writer.WriteLine($"with ({string.Join(", ", value.DimNames)}) as ({ivs}):");
        writer.Indent++;
        var set = value.Dependences.ToDictionary(d => d.Index, d => d.Node);
        for (int i = 0; i < value.ReadAccesses.Length; i++)
        {
            writer.Write($"read_{i} = ");
            writer.Write(value.ReadAccesses[i]);
            if (set.ContainsKey(i))
            {
                writer.WriteLine($" @ Op{set[i].OpId}");
            }
            else
            {
                writer.WriteLine();
            }
        }

        writer.Write("write = ");
        writer.WriteLine(value.WriteAccess);
        writer.Indent--;

        return default;
    }
}
