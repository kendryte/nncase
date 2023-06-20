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
            // 1. first simplify const folding.
            var visited = new Dictionary<ENode, bool>();
            foreach (var eclass in eGraph.Classes)
            {
                if (eclass.Nodes.Count > 1 && eclass.Nodes.Count(e => e.Expr is Const) == 1)
                {
                    foreach (var enode in eclass.Nodes.Where(e => e.Expr is not Const))
                    {
                        if (!visited.ContainsKey(enode))
                        {
                            cpmodel.AddAssumption(vars[enode].Not());
                            visited.Add(enode, true);
                        }
                    }
                }
            }

            EliminateAllCycles(root, new(), new(), visited, cpmodel, vars);
        }

        // 3. add pick weights for all enode.
        cpmodel.Minimize(LinearExpr.WeightedSum(eGraph.Nodes.Select(n => vars[n]), eGraph.Nodes.Select(n => checked((long)_costModel[n].Score))));

        if (cpmodel.Validate().Any())
        {
            throw new InvalidDataException("the sat model invalid: " + cpmodel.Validate());
        }

        var solver = new CpSolver();
        solver.StringParameters = $"max_time_in_seconds:{10f},num_workers:0";

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

    private void EliminateAllCycles(EClass root, LinkedList<(EClass Class, ENode Node)> path, Dictionary<EClass, LinkedListNode<(EClass Class, ENode Node)>> pathMemo, Dictionary<ENode, bool> visited, CpModel cpModel, Dictionary<ENode, BoolVar> vars)
    {
        // note how to avoid duplicate visit same cycle ?
        // simulate the extract, disable the all cycle path.
        // when detect the cycle, do not pick the cycle path
        if (pathMemo.TryGetValue(root, out _))
        {
            var (_, node) = path.Last!.Value;
            cpModel.AddAssumption(vars[node].Not());

            // var cycle = new List<BoolVar>();
            // do
            // {
            //     cycle.Add(vars[oldNode!.Value.Node]);
            //     oldNode = oldNode.Next;
            // } while (oldNode is not null);

            // if (cycle.Count == 1)
            // {
            //     // eg. eclass: [marker(x) , x], don't pick marker.
            //     cpModel.AddAssumption(cycle[0].Not());
            // }
            // else
            // {
            //     // note maybe we just do not pick backward node?
            //     cpModel.Add(cpModel.NewConstant(cycle.Count) != LinearExpr.Sum(cycle));
            // }
            return;
        }

        foreach (var enode in root.Nodes)
        {
            if (!visited.ContainsKey(enode))
            {
                foreach (var ch in enode.Children)
                {
                    var linkNode = path.AddLast((root, enode));
                    pathMemo.Add(root, linkNode);
                    EliminateAllCycles(ch, path, pathMemo, visited, cpModel, vars);
                    path.Remove(linkNode);
                    pathMemo.Remove(root);
                }

                visited.Add(enode, true);
            }
        }
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
