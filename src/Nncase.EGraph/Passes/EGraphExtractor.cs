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

namespace Nncase.Passes;

public delegate void EGraphExtractConstrains(CpModel model, IReadOnlyDictionary<ENode, BoolVar> vars);

internal class EGraphExtractor
{
    private readonly EGraphCostModel _costModel;

    public EGraphExtractor(EGraphCostModel costModel)
    {
        _costModel = costModel;
    }

    public Expr Extract(EClass root, IEGraph eGraph, EGraphExtractConstrains[] constrains)
    {
        var cpmodel = new CpModel();

        // 0. create bool var for all enode.
        var varMemo = new Dictionary<ENode, BoolVar>();
        foreach (var cls in eGraph.Classes)
        {
            foreach (var (e, i) in cls.Nodes.Select((e, i) => (e, i)))
            {
                varMemo.Add(e, cpmodel.NewBoolVar($"{cls.Id}_{i}"));
            }
        }

        // 1. must pick one in root enode.
        cpmodel.AddBoolOr(root.Nodes.Select(n => varMemo[n]).ToArray());

        // 2. when pick node, must pick one child node.
        foreach (var n in eGraph.Nodes)
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
            var class_cycles = FindCycles(hgraph);
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
        cpmodel.Minimize(LinearExpr.WeightedSum(eGraph.Nodes.Select(n => varMemo[n]), eGraph.Nodes.Select(n => checked((long)_costModel[n].Score))));

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

        var picks = eGraph.Nodes.ToDictionary(e => e, e => solver.BooleanValue(varMemo[e]));
        using (var dumpStream = enableDump ? DumpScope.Current.OpenFile("Costs/Pick.dot") : Stream.Null)
        {
            EGraphPrinter.DumpEgraphAsDot(eGraph, _costModel, picks, root.Find(), dumpStream);
        }

        return new SatExprBuildVisitor(picks).Visit(root);
    }

    private static HyperGraph ToHyperGraph(EClass root)
    {
        var hgraph = new HyperGraph();
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

    private static void SCCImpl(EClass v, HyperGraph graph, Dictionary<EClass, int> num, Dictionary<EClass, int> low, Stack<EClass> stack, HashSet<EClass> visited, HashSet<EClass> onstack, ref int idx, List<List<EClass>> scc)
    {
        num[v] = idx;
        low[v] = idx;
        idx++;
        visited.Add(v);
        stack.Push(v);
        onstack.Add(v);

        foreach (var u in graph.Neighbors(v))
        {
            if (!visited.Contains(u))
            {
                SCCImpl(u, graph, num, low, stack, visited, onstack, ref idx, scc);
                low[v] = Math.Min(low[v], low[u]);
            }
            else if (onstack.Contains(u))
            {
                low[v] = Math.Min(low[v], num[u]);
            }
        }

        if (low[v] == num[v])
        {
            var sccFound = new List<EClass>();
            EClass sccRt;
            do
            {
                sccRt = stack.Pop();
                onstack.Remove(sccRt);
                sccFound.Add(sccRt);
            } while (sccRt.Id != v.Id);
            scc.Add(sccFound);
        }
    }

    private static List<List<EClass>> FindSCC(HyperGraph graph)
    {
        var num = new Dictionary<EClass, int>();
        var low = new Dictionary<EClass, int>();
        var visited = new HashSet<EClass>();
        var processed = new HashSet<EClass>();
        var stack = new Stack<EClass>();
        int idx = 0;
        var scc = new List<List<EClass>>();

        foreach (var v in graph.Nodes().OrderBy(n => n.Id))
        {
            if (!visited.Contains(v))
            {
                SCCImpl(v, graph, num, low, stack, visited, processed, ref idx, scc);
            }
        }

        return scc;
    }

    private static void Unblock(EClass v, HashSet<EClass> blocked, Dictionary<EClass, HashSet<EClass>> blockedMap)
    {
        blocked.Remove(v);
        if (blockedMap.TryGetValue(v, out var blockedSet))
        {
            var worklist = blockedSet.ToList();
            foreach (var w in worklist)
            {
                if (blocked.Contains(w))
                {
                    Unblock(w, blocked, blockedMap);
                }
            }
        }
    }

    private static bool JohnsonAlgImpl(
        EClass s,
        EClass v,
        HyperGraph graph,
        HashSet<EClass> blocked,
        List<EClass> stack,
        Dictionary<EClass, HashSet<EClass>> blockMap,
        List<List<EClass>> cycles)
    {
        bool f = false;
        blocked.Add(v);
        stack.Add(v);

        foreach (var w in graph.Neighbors(v))
        {
            if (w.Equals(s))
            {
                f = true;
                cycles.Add(new List<EClass>(stack));
            }
            else if (!blocked.Contains(w))
            {
                f = JohnsonAlgImpl(s, w, graph, blocked, stack, blockMap, cycles) || f;
            }
        }

        if (f)
        {
            Unblock(v, blocked, blockMap);
        }
        else
        {
            foreach (var w in graph.Neighbors(v))
            {
                if (!blockMap.ContainsKey(w))
                {
                    blockMap[w] = new HashSet<EClass>();
                }

                blockMap[w].Add(v);
            }
        }

        stack.RemoveAt(stack.Count - 1);
        return f;
    }

    private static List<List<EClass>> FindCycles(HyperGraph hgraph)
    {
        var scc = FindSCC(hgraph)
            .Where(c => c.Count > 1)
            .ToList();

        var cycles = new List<List<EClass>>();
        foreach (var n in hgraph.Nodes())
        {
            if (hgraph.Neighbors(n).Contains(n))
            {
                cycles.Add(new List<EClass> { n });
            }
        }

        var blocked = new HashSet<EClass>();
        var blockMap = new Dictionary<EClass, HashSet<EClass>>();
        var stack = new List<EClass>();

        while (scc.Count > 0)
        {
            var curScc = scc[scc.Count - 1];
            scc.RemoveAt(scc.Count - 1);
            var subgraph = hgraph.SubGraph(curScc);

            for (int i = 0; i < curScc.Count; i++)
            {
                blocked.Clear();
                blockMap.Clear();
                var v = subgraph.GetIdByNode(i);
                JohnsonAlgImpl(v, v, subgraph, blocked, stack, blockMap, cycles);
                subgraph.RemoveNodeRaw(i);
            }
        }

        return cycles;
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
                expr = @if.With(condition: children[^3], then: children[^2], @else: children[^1], paramList: children[..^3].ToArray());
                break;
            default:
                throw new NotSupportedException(enode.Expr.GetType().Name);
        }

        _memo.Add(root, expr);

        return expr;
    }
}

internal sealed class HyperGraph
{
    private readonly Dictionary<int, Dictionary<int, HashSet<ENode>>> _edges;
    private readonly HashSet<int> _nodes;
    private readonly Dictionary<EClass, int> _ids_to_nodes;
    private readonly Dictionary<int, EClass> _nodes_to_ids;
    private int _num_nodes;

    public HyperGraph()
    {
        _edges = new();
        _nodes = new();
        _ids_to_nodes = new();
        _nodes_to_ids = new();
        _num_nodes = 0;
    }

    public bool Contains(EClass id)
    {
        return _ids_to_nodes.ContainsKey(id);
    }

    public Dictionary<EClass, HashSet<ENode>>? Edges(EClass eclass)
    {
        if (Contains(eclass))
        {
            var result = new Dictionary<EClass, HashSet<ENode>>();
            foreach (var (to, enodes) in _edges[_ids_to_nodes[eclass]])
            {
                result.Add(_nodes_to_ids[to], enodes);
            }

            return result;
        }

        return null;
    }

    public HashSet<EClass> Nodes()
    {
        return new(_nodes.Select(x => _nodes_to_ids[x]));
    }

    public void AddNode(EClass k)
    {
        var node_id = _num_nodes;
        _ids_to_nodes.Add(k, node_id);
        _nodes_to_ids.Add(node_id, k);
        _edges.Add(node_id, new());
        _nodes.Add(node_id);
        _num_nodes += 1;
    }

    public void Connect(EClass cfrom, EClass cto, ENode enode)
    {
        if (!Contains(cfrom))
        {
            AddNode(cfrom);
        }

        if (!Contains(cto))
        {
            AddNode(cto);
        }

        var from = _ids_to_nodes[cfrom];
        var to = _ids_to_nodes[cto];
        if (!_edges[from].ContainsKey(to))
        {
            _edges[from].Add(to, new HashSet<ENode>(new[] { enode }));
        }
        else
        {
            _edges[from][to].Add(enode);
        }
    }

    public EClass[] Neighbors(EClass u)
    {
        if (Contains(u))
        {
            return _edges[_ids_to_nodes[u]].Keys.Select(x => _nodes_to_ids[x]).ToArray();
        }

        return Array.Empty<EClass>();
    }

    public int GetNodeById(EClass id)
    {
        return _ids_to_nodes[id];
    }

    public EClass GetIdByNode(int node)
    {
        return _nodes_to_ids[node];
    }

    public void RemoveNodeRaw(int node)
    {
        if (_nodes.Contains(node))
        {
            _edges.Remove(node);
            foreach (var (_, v) in _edges)
            {
                v.Remove(node);
            }

            _nodes.Remove(node);
        }
    }

    public void RemoveNode(EClass node)
    {
        var node_id = _ids_to_nodes[node];
        if (Contains(node))
        {
            _edges.Remove(node_id);
            foreach (var (_, v) in _edges)
            {
                v.Remove(node_id);
            }

            _nodes.Remove(node_id);
        }
    }

    public int Size()
    {
        return _nodes.Count;
    }

    public HyperGraph SubGraph(IEnumerable<EClass> nodes)
    {
        var graph = new HyperGraph();
        var node_set = new HashSet<EClass>(nodes);
        foreach (var n in node_set)
        {
            var nedges = _edges[_ids_to_nodes[n]];
            foreach (var (neighbor, enodes) in nedges)
            {
                if (!node_set.Contains(_nodes_to_ids[neighbor]))
                {
                    continue;
                }

                foreach (var enode in enodes)
                {
                    graph.Connect(n, _nodes_to_ids[neighbor], enode);
                }
            }
        }

        return graph;
    }

    public void Dump(string name)
    {
        using (var dumpStream = DumpScope.Current.OpenFile($"Costs/{name}.dot"))
        {
            using var writer = new StreamWriter(dumpStream);
            foreach (var (u, v) in _edges)
            {
                foreach (var w in v.Keys)
                {
                    writer.WriteLine($"{_nodes_to_ids[u]} -> {_nodes_to_ids[w]}\n");
                }
            }
        }
    }
}
