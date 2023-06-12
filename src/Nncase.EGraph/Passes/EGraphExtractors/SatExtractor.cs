// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Google.OrTools.Sat;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase.Passes.EGraphExtractors;

internal class SatExtractor : IExtractor
{
    private readonly EGraphCostModel _costModel;

    public SatExtractor(EGraphCostModel costModel)
    {
        _costModel = costModel;
    }

    public Expr Extract(EClass root, IEGraph eGraph)
    {
        var cpmodel = new CpModel();

        // 0. create bool var for all enode.
        var vars = new Dictionary<ENode, BoolVar>();
        foreach (var item in eGraph.Nodes.Select((e, i) => (e, i)))
        {
            vars.Add(item.e, cpmodel.NewBoolVar(item.i.ToString()));
        }

        // 1. must pick one in root enode.
        cpmodel.AddBoolOr(root.Nodes.Select(n => vars[n]).ToArray());

        // 2. when pick node, must pick one child node.
        foreach (var n in eGraph.Nodes)
        {
            var ns = new[] { vars[n].Not() };
            foreach (var child in n.Children)
            {
                cpmodel.AddBoolOr(ns.Concat(child.Nodes.Select(cn => vars[cn])));
            }
        }

        // 3. no cycle
        {
            EliminateAllCycles(root, new(), new(), cpmodel, vars);
            // int cycleVarCount = 0;
            // GetAllCycles(root, new Dictionary<EClass, int>(), new List<(EClass Class, ENode Node)>(), cpmodel, vars, ref cycleVarCount);
        }

        // 3. add pick weights for all enode.
        cpmodel.Minimize(LinearExpr.WeightedSum(eGraph.Nodes.Select(n => vars[n]), eGraph.Nodes.Select(n => checked((long)_costModel[n].Score))));

        if (cpmodel.Validate().Any())
        {
            throw new InvalidDataException("the sat model invalid: " + cpmodel.Validate());
        }

        var solver = new CpSolver();

        var enableDump = DumpScope.Current.IsEnabled(DumpFlags.EGraphCost);
        CpSolverStatus status;
        using (var dumpStream = enableDump ? DumpScope.Current.OpenFile("Costs/Solve.txt") : Stream.Null)
        {
            using var writer = new StreamWriter(dumpStream);
            var cb = new PrintCostCallBack(vars, _costModel, writer, enableDump);
            status = solver.Solve(cpmodel, cb);
            dumpStream.Flush();
        }

        if (status is not (CpSolverStatus.Optimal or CpSolverStatus.Feasible))
        {
            throw new InvalidProgramException("SatExtract Failed!");
        }

        var pick = eGraph.Nodes.ToDictionary(e => e, e => solver.BooleanValue(vars[e]));
        using (var dumpStream = enableDump ? DumpScope.Current.OpenFile("Costs/Pick.dot") : Stream.Null)
        {
            EGraphPrinter.DumpEgraphAsDot(eGraph, _costModel, pick, root.Find(), dumpStream);
        }

        return new SatExprBuildVisitor(pick).Visit(root);
    }

    private void EliminateAllCycles(EClass root, LinkedList<(EClass Class, ENode Node)> path, Dictionary<EClass, LinkedListNode<(EClass Class, ENode Node)>> visited, CpModel cpModel, Dictionary<ENode, BoolVar> vars)
    {
        // note how to avoid duplicate visit same cycle ?
        // simulate the extract, disable the all cycle path.
        // when detect the cycle, do not pick the cycle path
        if (visited.TryGetValue(root, out var oldNode))
        {
            var cycle = new List<BoolVar>();
            do
            {
                cycle.Add(vars[oldNode!.Value.Node]);
                oldNode = oldNode.Next;
            } while (oldNode is not null);

            if (cycle.Count == 1)
            {
                // eg. eclass: [marker(x) , x], don't pick marker.
                cpModel.AddAssumption(cycle[0].Not());
            }
            else
            {
                // note maybe we just do not pick backward node?
                cpModel.Add(cpModel.NewConstant(cycle.Count) != LinearExpr.Sum(cycle));
            }

            return;
        }

        foreach (var enode in root.Nodes)
        {
            foreach (var ch in enode.Children)
            {
                var linkNode = path.AddLast((root, enode));
                visited.Add(root, linkNode);
                EliminateAllCycles(ch, path, visited, cpModel, vars);
                path.Remove(linkNode);
                visited.Remove(root);
            }
        }
    }

    private void GetAllCycles(EClass root, Dictionary<EClass, int> visited, List<(EClass Class, ENode Node)> path, CpModel cpModel, IReadOnlyDictionary<ENode, BoolVar> vars, ref int cycleVarCount)
    {
        if (visited.ContainsKey(root) && visited[root] == 2)
        {
            return;
        }

        if (visited.ContainsKey(root) && visited[root] == 1)
        {
            if (path.FindIndex(p => p.Class == root) is int idx && idx != -1)
            {
                var new_cycle = new List<BoolVar>();
                var subpath = path.Skip(idx).ToArray();
                foreach (var (_, n) in subpath)
                {
                    new_cycle.Add(vars[n]);
                }

                if (new_cycle.Count == 1)
                {
                    // eg. eclass: [marker(x) , x], don't pick marker.
                    cpModel.AddAssumption(new_cycle[0].Not());
                }
                else
                {
                    // EncodeCycle(subpath, cpModel, vars, ref cycleVarCount);
                    cpModel.Add(cpModel.NewConstant(subpath.Length) != LinearExpr.Sum(subpath.Select(p => vars[p.Node])));
                }

                return;
            }

            throw new InvalidOperationException($"Should have a cycle here: {root} , {path}.");
        }

        visited[root] = 1;
        foreach (var node in root.Nodes)
        {
            foreach (var ch in node.Children)
            {
                path.Add((root, node));
                GetAllCycles(ch, visited, path, cpModel, vars, ref cycleVarCount);
                path.RemoveAt(path.Count - 1);
            }
        }

        visited[root] = 2;
    }

    private void EncodeCycle(ReadOnlySpan<(EClass Class, ENode Node)> subpath, CpModel cpModel, IReadOnlyDictionary<ENode, BoolVar> vars, ref int cycleVarCount)
    {
        var clauses = AggregateClauses(subpath, cpModel, vars);
        TseytinEncoding(clauses, cpModel, ref cycleVarCount);
    }

    private List<List<BoolVar>> AggregateClauses(ReadOnlySpan<(EClass Class, ENode Node)> subpath, CpModel cpModel, IReadOnlyDictionary<ENode, BoolVar> vars)
    {
        var clauses = new List<List<BoolVar>>();
        for (int i = 0; i < subpath.Length; i++)
        {
            var clause = new List<BoolVar>();
            foreach (var var in subpath[i].Class.Nodes.Select(x => vars[x]))
            {
                clause.Add(var);
            }

            clauses.Add(clause);
        }

        return clauses;
    }

    private void TseytinEncoding(List<List<BoolVar>> clauses, CpModel cpModel, ref int cycleVarCount)
    {
        var var_map = new Dictionary<int, BoolVar>();
        for (int i = 0; i < clauses.Count; i++)
        {
            var c = clauses[i];
            if (c.Count > 1)
            {
                // new variable to represent the clause
                var v = cpModel.NewBoolVar($"cycle_v_{cycleVarCount++}");
                var_map.Add(i, v);

                // v <-> c
                // == v -> c /\ c -> v
                // == -v \/ c /\ -c \/ v
                // == -v \/ c AND -c \/ v
                // for `c`, it is a conjunction of (negation of) variables therefore
                // 1. -v \/ c == -v \/ -x /\ -v \/ -y /\ -v \/ -z ...
                // -c \/ v == -(-x /\ -y /\ -z ...) \/ v
                // 2. == x \/ y \/ z \/ ... \/ v

                // Add 1 as hard clauses
                foreach (var x in c)
                {
                    cpModel.AddBoolOr(new[] { v.Not(), x.Not() });
                }

                // Add 2 as hard clauses
                cpModel.AddBoolOr(c.Concat(new[] { v }));
            }
        }

        // Finally, tseytin encoding for the clauses
        // == v1 \/ v2 \/ ... \/ vn
        cpModel.AddBoolOr(clauses.Select((c, i) => c.Count > 1 ? var_map[i] : c[0].Not()).ToArray());
    }
}

internal sealed class PrintCostCallBack : CpSolverSolutionCallback
{
    private readonly IReadOnlyDictionary<ENode, BoolVar> _vars;
    private readonly EGraphCostModel _costModel;
    private readonly StreamWriter _dumpWriter;
    private readonly bool _enableDump;
    private int _count;

    public PrintCostCallBack(IReadOnlyDictionary<ENode, BoolVar> vars, EGraphCostModel costModel, StreamWriter writer, bool enableDump)
    {
        _vars = vars;
        _costModel = costModel;
        _dumpWriter = writer;
        _enableDump = enableDump;
    }

    public override void OnSolutionCallback()
    {
        if (_enableDump)
        {
            var cost = Cost.Zero;
            foreach (var (n, v) in _vars)
            {
                if (_costModel[n] != Cost.Zero && BooleanValue(v))
                {
                    cost += _costModel[n];
                }
            }

            _dumpWriter.WriteLine($"Solution {_count++} @ {WallTime()}:");
            _dumpWriter.WriteLine(cost.ToString());
        }
    }
}

internal sealed class SatExprBuildVisitor
{
    private readonly IReadOnlyDictionary<ENode, bool> _pick;
    private readonly Dictionary<EClass, Expr> _memo;

    public SatExprBuildVisitor(IReadOnlyDictionary<ENode, bool> pick)
    {
        _pick = pick;
        _memo = new();
    }

    public Expr Visit(EClass root)
    {
        Expr? expr;
        if (_memo.TryGetValue(root, out expr))
        {
            return expr;
        }

        var enodes = root.Nodes.Where(n => _pick[n]).ToArray();
        if (enodes.Length != 1)
        {
            throw new InvalidProgramException("the one eclass only can pick one enode!");
        }

        var enode = enodes[0];
        var children = enode.Children.Select(e => Visit(e)).ToArray();

        switch (enode.Expr)
        {
            case Var or TensorConst or TupleConst or Op or Fusion or None:
                expr = enode.Expr;
                break;
            case Function func:
                expr = func.With(body: children[0], parameters: children[1..].OfType<Var>().ToArray());
                break;
            case Call call:
                expr = call.With(target: children[0], arguments: children[1..], call.Metadata);
                break;
            case IR.Tuple tp:
                expr = tp.With(fields: children);
                break;
            case Marker mk:
                expr = mk.With(target: children[0], attribute: children[1]);
                break;
            case IR.If @if:
                expr = @if.With(condition: children[0], then: children[1], @else: children[2]);
                break;
            default:
                throw new NotSupportedException(enode.Expr.GetType().Name);
        }

        _memo.Add(root, expr);

        return expr;
    }
}
