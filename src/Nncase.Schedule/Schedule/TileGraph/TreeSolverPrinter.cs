// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.CodeDom.Compiler;
using System.Reactive;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileGraph;

public sealed class TreeSolverPrinter : TreeSolverBase<IntExpr>, ITreeNodeVisitor<IndentedTextWriter, Unit>
{
    public TreeSolverPrinter(Assignment? solution, Solver solver, Dictionary<OpNode, OpNodeInfo<IntExpr>> opNodeMemo, Dictionary<TileNode, TileNodeInfo<IntExpr>> tileNodeMemo, Dictionary<ITileable, DomainInfo<IntExpr>> tileableNodeMemo, ICpuTargetOptions targetOptions)
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

            writer.WriteLine($"- {i}: {intExprs[i].ToSimplifyString()} {value}");
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
            WriteIntExprVector(writer, $"- {i}", vector, solution);
        }

        writer.Indent--;
    }

    public Unit Visit(TileNode value, IndentedTextWriter writer)
    {
        writer.WriteLine($"TileNode {value}:");
        writer.Indent++;

        writer.WriteLine($"Domain Relation: {value.DomainRelation}");
        WriteIntExprVector(writer, "TileVars", TileableNodeMemo[value].TileVars, Solution);

        WriteIntExprVector(writer, "TripCounts", TileNodeMemo[value].TripCounts, Solution);
        WriteIntExprMatrix(writer, "BackWardExtents", TileNodeMemo[value].BackWardExtents, Solution);

        writer.WriteLine($"BufferInfo:");
        foreach (var (bid, info) in TileNodeMemo[value].BufferInfoMap)
        {
            writer.Indent++;
            writer.WriteLine($"{bid}:");
            {
                writer.Indent++;
                WriteIntExprMatrix(writer, "Shapes", info.Shapes, Solution);
                WriteIntExprVector(writer, "SizeVars", info.SizeVars, Solution);
                WriteIntExprVector(writer, "SizeExprs", info.SizeExprs, Solution);
                writer.Indent--;
            }

            writer.Indent--;
        }

        writer.WriteLine($"Children:");
        foreach (var item in value.Children)
        {
            writer.Indent++;
            item.Accept(this, writer);
            writer.Indent--;
        }

        writer.Indent--;

        return default;
    }

    public Unit Visit(OpNode value, IndentedTextWriter writer)
    {
        writer.WriteLine($"OpNode {value}:");
        writer.Indent++;

        writer.WriteLine($"Domain Relation: {value.DomainRelation}");
        WriteIntExprMatrix(writer, "Shapes", OpNodeMemo[value].Shapes, Solution);

        writer.Indent++;
        for (int i = 0; i < value.ReadAccesses.Length; i++)
        {
            writer.Write($"- read_{i}: ");
            writer.WriteLine(value.ReadAccesses[i]);
        }

        writer.Write("- write: ");
        writer.WriteLine(value.WriteAccess);
        writer.Indent--;
        writer.Indent--;

        return default;
    }
}
