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
using Nncase.Graphs;
using Nncase.IR;

namespace Nncase.Passes;

public delegate void EGraphExtractConstrains(CpModel model, IReadOnlyDictionary<ENode, BoolVar> vars);

internal class EGraphExtractor
{
    private readonly EGraphCostModel _costModel;

    public EGraphExtractor(EGraphCostModel costModel)
    {
        _costModel = costModel;
    }

    public BaseExpr Extract(EClass root, IEGraph eGraph, EGraphExtractConstrains[] constrains)
    {
        var cpmodel = new CpModel();
        var nodes = CollectNodes(root);

        // 0. create bool var for all enode.
        var varMemo = new Dictionary<ENode, BoolVar>();
        foreach (var cls in eGraph.Classes)
        {
            foreach (var (e, i) in cls.Nodes.Select((e, i) => (e, i)))
            {
                if (nodes.Contains(e))
                {
                    varMemo.Add(e, cpmodel.NewBoolVar($"{cls.Id}_{i}"));
                }
            }
        }

        // 1. must pick one in root enode.
        cpmodel.AddBoolOr(root.Nodes.Select(n => varMemo[n]).ToArray());

        // 2. when pick node, must pick one child node.
        foreach (var n in nodes)
        {
            var ns = new[] { varMemo[n].Not() };
            foreach (var child in n.Children)
            {
                cpmodel.AddBoolOr(ns.Concat(child.Nodes.Select(cn => varMemo[cn])));
            }
        }

        // 3. no cycle
        {
            var hgraph = ToHyperGraph(root);
            var class_cycles = hgraph.FindCycles();
            foreach (var cycle in class_cycles)
            {
                if (cycle.Count == 1)
                {
                    foreach (var n in cycle[0].Nodes)
                    {
                        if (n.Children.Contains(cycle[0]))
                        {
                            cpmodel.AddAssumption(varMemo[n].Not());
                        }
                    }
                }
                else
                {
                    // build clauses.
                    var clauses = new List<List<BoolVar>>();
                    for (int i = 0; i < cycle.Count; i++)
                    {
                        var next_hop = (i + 1) % cycle.Count;
                        var u = hgraph.Edges(cycle[i])!;
                        var v = u[cycle[next_hop]];
                        clauses.Add(v.Select(n => varMemo[n]).ToList());
                    }

                    var clauseMemo = new Dictionary<int, BoolVar>();
                    for (int i = 0; i < clauses.Count; i++)
                    {
                        var clause = clauses[i];
                        if (clause.Count > 1)
                        {
                            var tmpV = cpmodel.NewBoolVar(string.Empty);
                            cpmodel.AddBoolAnd(clause.Select(c => c.Not())).OnlyEnforceIf(tmpV);
                            cpmodel.AddBoolOr(clause).OnlyEnforceIf(tmpV.Not());
                            clauseMemo.Add(i, tmpV);
                        }
                    }

                    cpmodel.AddBoolOr(clauses.Select((c, i) => (c, i)).Select(p => p.c.Count == 1 ? p.c[0].Not() : clauseMemo[p.i]));
                }
            }
        }

        foreach (var constrain in constrains)
        {
            constrain(cpmodel, varMemo);
        }

        // 3. add pick weights for all enode.
        cpmodel.Minimize(LinearExpr.WeightedSum(nodes.Select(n => varMemo[n]), nodes.Select(n => checked((long)_costModel[n].Score))));

        if (cpmodel.Validate().Any())
        {
            throw new InvalidDataException("the sat model invalid: " + cpmodel.Validate());
        }

        var solver = new CpSolver();
        int max_time = 120;
        if (System.Environment.GetEnvironmentVariable("SOLVE_MAX_TIME") is string s_solve_max_time)
        {
            try
            {
                var solve_max_time = int.Parse(s_solve_max_time);
                max_time = solve_max_time;
            }
            catch (System.Exception)
            {
            }
        }

        int processorCount = Math.Max(System.Environment.ProcessorCount / 2, 1);
        if (System.Environment.GetEnvironmentVariable("SOLVE_PROCESSOR_COUNT") is string s_solve_processor_count)
        {
            try
            {
                var solve_processor_count = int.Parse(s_solve_processor_count);
                processorCount = solve_processor_count;
            }
            catch (System.Exception)
            {
            }
        }

        solver.StringParameters = $"max_time_in_seconds:{max_time},num_workers:{processorCount}";

        var enableDump = DumpScope.Current.IsEnabled(DumpFlags.EGraphCost);
        CpSolverStatus status;
        using (var dumpStream = enableDump ? DumpScope.Current.OpenFile("Costs/Solve.txt") : Stream.Null)
        {
            using var writer = new StreamWriter(dumpStream);
            var cb = new PrintCostCallBack(varMemo, _costModel, writer, enableDump);
            status = solver.Solve(cpmodel, cb);
            writer.WriteLine($"Status : {status}");
            dumpStream.Flush();
        }

        if (status is not (CpSolverStatus.Optimal or CpSolverStatus.Feasible))
        {
            throw new InvalidProgramException("SatExtract Failed!");
        }

        var picks = nodes.ToDictionary(e => e, e => solver.BooleanValue(varMemo[e]));
        using (var dumpStream = enableDump ? DumpScope.Current.OpenFile("Costs/Pick.dot") : Stream.Null)
        {
            EGraphPrinter.DumpEgraphAsDot(eGraph, _costModel, picks, root.Find(), dumpStream);
        }

        return new SatExprBuildVisitor(picks).Visit(root);
    }

    private static HyperGraph<EClass, ENode> ToHyperGraph(EClass root)
    {
        var hgraph = new HyperGraph<EClass, ENode>();
        var visited = new HashSet<EClass>();
        var queue = new Queue<EClass>();
        queue.Enqueue(root);
        visited.Add(root);
        while (queue.Any())
        {
            var front = queue.Dequeue();
            foreach (var node in front.Nodes)
            {
                foreach (var ch in node.Children)
                {
                    var canonical = ch;
                    hgraph.Connect(front, canonical, node);
                    if (!visited.Contains(canonical))
                    {
                        visited.Add(canonical);
                        queue.Enqueue(canonical);
                    }
                }
            }
        }

        return hgraph;
    }

    private HashSet<ENode> CollectNodes(EClass root)
    {
        var visited = new HashSet<ENode>();
        void Visit(ENode node)
        {
            if (visited.Add(node))
            {
                foreach (var child in node.Children)
                {
                    foreach (var n in child.Nodes)
                    {
                        Visit(n);
                    }
                }
            }
        }

        foreach (var n in root.Nodes)
        {
            Visit(n);
        }

        return visited;
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
            _dumpWriter.Flush();
        }
    }
}

internal sealed class SatExprBuildVisitor
{
    private readonly IReadOnlyDictionary<ENode, bool> _pick;
    private readonly Dictionary<EClass, BaseExpr> _memo;

    public SatExprBuildVisitor(IReadOnlyDictionary<ENode, bool> pick)
    {
        _pick = pick;
        _memo = new();
    }

    public BaseExpr Visit(EClass root)
    {
        BaseExpr? expr;
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
            case Var or TensorConst or TupleConst or Op or Fusion or None or Dimension or Shape:
                expr = enode.Expr;
                break;
            case Function func:
                expr = children.Length == 0 ? func : func.With(body: (Expr)children[0], parameters: children[1..].OfType<Var>().ToArray());
                break;
            case If @if:
                expr = @if.With(condition: (Expr)children[0], then: (BaseFunction)children[1], @else: (BaseFunction)children[2], arguments: children[3..]);
                break;
            case Call call:
                expr = call.With(target: (Expr)children[0], arguments: children[1..], call.Metadata);
                break;
            case IR.Tuple tp:
                expr = tp.With(fields: children);
                break;
            case Marker mk:
                expr = mk.With(target: (Expr)children[0], attribute: children[1], metadata: mk.Metadata);
                break;
            default:
                throw new NotSupportedException(enode.Expr.GetType().Name);
        }

        _memo.Add(root, expr);

        return expr;
    }
}
